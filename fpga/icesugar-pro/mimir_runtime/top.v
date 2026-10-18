module top #(
    parameter integer CLOCK_HZ = 25_000_000,
    parameter integer BAUD = 115_200,
    parameter integer REQUEST_TIMEOUT_CYCLES = CLOCK_HZ / 10,
    parameter integer LED_FAST_CYCLES = CLOCK_HZ / 12,
    parameter integer LED_SLOW_CYCLES = CLOCK_HZ / 2,
    parameter integer LED_LEASE_CYCLES = CLOCK_HZ * 5
) (
    input wire clk25,
    input wire serial_rx,
    output wire serial_tx,
    (* LOC="A11", IO_TYPE="LVCMOS33" *) output wire led_green,
    (* LOC="B11", IO_TYPE="LVCMOS33" *) output wire led_red,
    (* LOC="A12", IO_TYPE="LVCMOS33" *) output wire led_blue
);
    localparam integer CLOCKS_PER_BIT = CLOCK_HZ / BAUD;
    localparam integer FIFO_DEPTH = 32;

    reg [15:0] rx_clock = 0;
    reg [3:0] rx_bit = 0;
    reg [7:0] rx_shift = 0;
    reg rx_busy = 0;
    reg rx_ready = 0;
    reg rx_meta = 1;
    reg rx_sync = 1;

    reg [15:0] tx_clock = 0;
    reg [3:0] tx_bit = 0;
    reg [9:0] tx_shift = 10'b11_1111_1111;
    reg tx_busy = 0;
    reg tx_line = 1;

    reg [7:0] tx_fifo [0:FIFO_DEPTH - 1];
    reg [4:0] fifo_write = 0;
    reg [4:0] fifo_read = 0;
    reg [5:0] fifo_count = 0;

    reg [$clog2(REQUEST_TIMEOUT_CYCLES + 1)-1:0] request_idle = 0;
    reg matrix_valid = 0;
    // Manufacturer's iCESugar-Pro RGB pins are active-high.
    // Mapping is kept here deliberately, so the existing LPF stays unchanged.
    wire led_on, led_active;
    // Explicit pad cells preserve three separate LOC attributes even though
    // the white indicator drives all three channels with the same signal.
    // Match the existing LPF's 3.3 V UART on the shared I/O bank.
    (* LOC="A11", IO_TYPE="LVCMOS33" *) TRELLIS_IO #(.DIR("OUTPUT")) green_pad
        (.B(led_green), .I(led_on), .T(1'b0), .O());
    (* LOC="B11", IO_TYPE="LVCMOS33" *) TRELLIS_IO #(.DIR("OUTPUT")) red_pad
        (.B(led_red), .I(led_on), .T(1'b0), .O());
    (* LOC="A12", IO_TYPE="LVCMOS33" *) TRELLIS_IO #(.DIR("OUTPUT")) blue_pad
        (.B(led_blue), .I(led_on), .T(1'b0), .O());

    reg [3:0] caps_match = 0;
    reg [2:0] dot_match = 0;
    reg [2:0] load_match = 0;
    reg [2:0] exec_match = 0;
    reg [2:0] matrix_load_match = 0;
    reg [2:0] matrix_exec_match = 0;
    reg [1:0] dot_state = 0;
    reg [6:0] dot_length = 0;
    reg [6:0] dot_index = 0;
    reg signed [7:0] dot_lhs = 0;
    reg signed [31:0] dot_accumulator = 0;
    reg signed [31:0] dot_result = 0;

    reg signed [7:0] resident_weights [0:63];
    reg [1:0] load_state = 0;
    reg [6:0] load_length = 0;
    reg [6:0] load_index = 0;
    reg [6:0] resident_length = 0;
    reg [1:0] exec_state = 0;
    reg [6:0] exec_length = 0;
    reg [6:0] exec_index = 0;
    reg signed [31:0] exec_accumulator = 0;

    reg signed [7:0] matrix_weights [0:511];
    reg signed [31:0] matrix_accumulators [0:7];
    reg signed [31:0] matrix_results [0:7];
    reg [1:0] matrix_load_state = 0;
    reg [3:0] matrix_rows = 0;
    reg [6:0] matrix_columns = 0;
    reg [9:0] matrix_load_index = 0;
    reg [1:0] matrix_exec_state = 0;
    reg [6:0] matrix_exec_index = 0;
    integer matrix_row;

    reg response_active = 0;
    reg [2:0] response_kind = 0;
    reg [9:0] response_index = 0;
    reg [7:0] response_byte = 0;

    // General I/O staging: bounded BRAM chunks, never a full-dataset cache.
    reg [7:0] io_buffer [0:511];
    reg [2:0] io_match = 0;
    reg [31:0] io_command = 0;
    reg [3:0] io_state = 0;
    reg [7:0] io_mode = 0;
    reg [8:0] io_length = 0, io_index = 0;
    reg [9:0] io_output_length = 0, io_crc_index = 0;
    reg [31:0] io_word = 0, io_expected_crc = 0;
    reg [31:0] io_write_word = 0;
    reg [2:0] io_write_count = 0;
    reg io_crc_ready = 0;
    reg [7:0] io_read_value;

    reg [31:0] io_input_crc = 32'hffffffff, io_output_crc = 32'hffffffff;
    reg io_bad = 0, checkpoint_begin_pending = 0;
    reg [7:0] checkpoint_operation = 0;
    reg [63:0] checkpoint_bytes = 0;
    reg [31:0] checkpoint_crc = 32'hffffffff;
    wire [31:0] io_complete_f32 = {rx_shift, io_word[23:0]};
    wire [15:0] io_complete_f16 = {rx_shift, io_word[7:0]};
    wire [15:0] io_encoded = io_mode == 1 ? f32_to_bf16(io_complete_f32) : f32_to_f16(io_complete_f32);
    wire [31:0] io_decoded = io_mode == 2 ? {io_complete_f16,16'b0} : f16_to_f32(io_complete_f16);

    function [31:0] crc_byte;
        input [31:0] crc;
        input [7:0] value;
        reg [31:0] next;
        integer bit_number;
        begin
            next = crc ^ value;
            for(bit_number=0;bit_number<8;bit_number=bit_number+1)
                next = next[0] ? (next >> 1) ^ 32'hedb88320 : next >> 1;
            crc_byte = next;
        end
    endfunction
    function [15:0] f32_to_bf16;
        input [31:0] value;
        reg [31:0] rounded;
        begin
            rounded = value + 32'h7fff + value[16];
            if(value[30:23]==255 && value[22:0]!=0) f32_to_bf16=value[31:16] | 16'h0040;
            else f32_to_bf16=rounded[31:16];
        end
    endfunction
    function [15:0] f32_to_f16;
        input [31:0] value;
        integer exponent, shift;
        reg [31:0] mantissa, remainder, halfway, rounded;
        reg [15:0] sign_bit;
        begin
            sign_bit={value[31],15'b0}; exponent=value[30:23]-112;
            mantissa={1'b1,value[22:0]}; rounded=0; remainder=0; halfway=0; shift=0;
            if(value[30:23]==255)
                f32_to_f16=sign_bit | 16'h7c00 | (value[22:0]!=0 ? ((value[22:13]) | 16'h0200) : 16'b0);
            else if(exponent>=31) f32_to_f16=sign_bit | 16'h7c00;
            else if(exponent < -10) f32_to_f16=sign_bit;
            else if(exponent<=0) begin
                shift=14-exponent; rounded=mantissa >> shift;
                remainder=mantissa & ((32'b1 << shift)-1); halfway=32'b1 << (shift-1);
                if(remainder>halfway || (remainder==halfway && rounded[0])) rounded=rounded+1;
                f32_to_f16=sign_bit | rounded[15:0];
            end else begin
                rounded=(exponent << 10) | (value[22:0] >> 13);
                if(value[12:0]>4096 || (value[12:0]==4096 && rounded[0])) rounded=rounded+1;
                f32_to_f16=sign_bit | rounded[15:0];
            end
        end
    endfunction
    function [31:0] f16_to_f32;
        input [15:0] value;
        integer exponent, n;
        reg [10:0] mantissa;
        reg [31:0] sign_bit;
        begin
            sign_bit={value[15],31'b0}; mantissa={1'b0,value[9:0]}; exponent=-14;
            if(value[14:10]==0) begin
                if(mantissa==0) f16_to_f32=sign_bit;
                else begin
                    for(n=0;n<10;n=n+1) if(!mantissa[10]) begin mantissa=mantissa << 1; exponent=exponent-1; end
                    f16_to_f32=sign_bit | ((exponent+127) << 23) | ((mantissa & 1023) << 13);
                end
            end else if(value[14:10]==31)
                f16_to_f32=sign_bit | 32'h7f800000 | (mantissa << 13) | (mantissa!=0 ? 32'h00400000 : 0);
            else f16_to_f32=sign_bit | ((value[14:10]+112) << 23) | (mantissa << 13);
        end
    endfunction

    wire request_pending = dot_state != 0 || load_state != 0 || exec_state != 0 ||
        matrix_load_state != 0 || matrix_exec_state != 0 || caps_match != 0 ||
        dot_match != 0 || load_match != 0 || exec_match != 0 ||
        matrix_load_match != 0 || matrix_exec_match != 0 || io_match != 0 || io_state != 0 || checkpoint_begin_pending;

    wire fifo_pop = !tx_busy && fifo_count != 0;
    wire fifo_push = response_active && (fifo_count != FIFO_DEPTH || fifo_pop);

    wire [8:0] io_read_address = io_state==9 ?
        ((io_crc_ready && io_crc_index<io_output_length)?io_crc_index+1'b1:io_crc_index) :
        (fifo_push ? response_index+1'b1-7 : response_index-7);

    wire led_start = fifo_push && response_index == 0 && response_kind == 0 && !led_active;
    wire led_keepalive = fifo_push && response_index == 0 && !led_start;
    mimir_status_led #(.FAST_CYCLES(LED_FAST_CYCLES), .SLOW_CYCLES(LED_SLOW_CYCLES),
        .LEASE_CYCLES(LED_LEASE_CYCLES)) status_led(
        clk25, led_start, led_keepalive, 1'b0, led_on, led_active);

    assign serial_tx = tx_line;

    always @(*) begin
        response_byte = 0;
        if (!response_active) begin
            response_byte = 0;
        end else if (response_kind == 0) begin
            case (response_index)
                0: response_byte = "M";
                1: response_byte = "I";
                2: response_byte = "M";
                3: response_byte = "I";
                4: response_byte = "R";
                5: response_byte = "F";
                6: response_byte = "P";
                7: response_byte = "G";
                8: response_byte = 8'd1;
                9: response_byte = 8'd3;
                10: response_byte = 8'd15;
                11, 12, 13: response_byte = 8'd0;
                14: response_byte = 8'd64;
                default: response_byte = 8'd0;
            endcase
        end else if (response_kind == 1) begin
            case (response_index)
                0: response_byte = "D";
                1: response_byte = "R";
                2: response_byte = "E";
                3: response_byte = "S";
                4: response_byte = dot_result[7:0];
                5: response_byte = dot_result[15:8];
                6: response_byte = dot_result[23:16];
                default: response_byte = dot_result[31:24];
            endcase
        end else if (response_kind == 2) begin
            case (response_index)
                0: response_byte = "W";
                1: response_byte = "A";
                2: response_byte = "C";
                default: response_byte = "K";
            endcase
        end else if (response_kind == 3) begin
            if (response_index == 0) response_byte = "M";
            else if (response_index == 1) response_byte = "R";
            else if (response_index == 2) response_byte = "E";
            else if (response_index == 3) response_byte = "S";
            else if (response_index == 4) response_byte = matrix_rows;
            else begin
                case ((response_index - 5) & 3)
                    0: response_byte = matrix_results[(response_index - 5) >> 2][7:0];
                    1: response_byte = matrix_results[(response_index - 5) >> 2][15:8];
                    2: response_byte = matrix_results[(response_index - 5) >> 2][23:16];
                    default: response_byte = matrix_results[(response_index - 5) >> 2][31:24];
                endcase
            end
        end else if(response_kind==5) begin
            case(response_index)
                0: response_byte="X";
                1: response_byte="R";
                2: response_byte="E";
                3: response_byte="S";
                4: response_byte=io_bad ? 255 : io_mode;
                5: response_byte=io_output_length[7:0];
                6: response_byte={6'b0,io_output_length[9:8]};
                default: begin
                    if(response_index < 7+io_output_length) response_byte=io_read_value;
                    else response_byte=(~io_output_crc) >> ((response_index-7-io_output_length)*8);
                end
            endcase
        end else if(response_kind==6) begin
            case(response_index)
                0: response_byte="C";
                1: response_byte="A";
                2: response_byte="C";
                default: response_byte="K";
            endcase
        end else if(response_kind==7) begin
            case(response_index)
                0: response_byte="C";
                1: response_byte="R";
                2: response_byte="E";
                3: response_byte="S";
                4: response_byte=checkpoint_operation;
                default: begin
                    if(response_index<13) response_byte=checkpoint_bytes >> ((response_index-5)*8);
                    else response_byte=(~checkpoint_crc) >> ((response_index-13)*8);
                end
            endcase
        end else begin
            case (response_index)
                0: response_byte = "M";
                1: response_byte = "A";
                2: response_byte = "C";
                default: response_byte = "K";
            endcase
        end
    end

    always @(posedge clk25) begin
        rx_meta <= serial_rx;
        rx_sync <= rx_meta;
        rx_ready <= 0;

        if (!rx_busy) begin
            if (!rx_sync) begin
                rx_busy <= 1;
                rx_clock <= CLOCKS_PER_BIT / 2;
                rx_bit <= 0;
            end
        end else if (rx_clock == 0) begin
            rx_clock <= CLOCKS_PER_BIT - 1;
            if (rx_bit == 0) begin
                if (rx_sync) rx_busy <= 0;
                else rx_bit <= 1;
            end else if (rx_bit <= 8) begin
                rx_shift[rx_bit - 1] <= rx_sync;
                rx_bit <= rx_bit + 1;
            end else begin
                rx_busy <= 0;
                if (rx_sync) rx_ready <= 1;
            end
        end else begin
            rx_clock <= rx_clock - 1'b1;
        end

        if (rx_ready && dot_state == 0 && load_state == 0 && exec_state == 0 &&
            matrix_load_state == 0 && matrix_exec_state == 0 && io_state == 0 && !checkpoint_begin_pending && !response_active) begin
            case (caps_match)
                0: caps_match <= (rx_shift == "M") ? 1 : 0;
                1: caps_match <= (rx_shift == "I") ? 2 : 0;
                2: caps_match <= (rx_shift == "M") ? 3 : 0;
                3: caps_match <= (rx_shift == "I") ? 4 : 0;
                4: caps_match <= (rx_shift == "R") ? 5 : 0;
                5: caps_match <= (rx_shift == "?") ? 6 : 0;
                6: caps_match <= (rx_shift == 8'd1) ? 7 : 0;
                7: begin
                    caps_match <= 0;
                    if (rx_shift == 8'h0a) begin
                        response_active <= 1;
                        response_kind <= 0;
                        response_index <= 0;
                    end
                end
            endcase

            case (dot_match)
                0: dot_match <= (rx_shift == "D") ? 1 : 0;
                1: dot_match <= (rx_shift == "O") ? 2 : 0;
                2: dot_match <= (rx_shift == "T") ? 3 : 0;
                3: begin
                    dot_match <= 0;
                    if (rx_shift == "8") dot_state <= 1;
                end
                default: dot_match <= 0;
            endcase
            case (load_match)
                0: load_match <= (rx_shift == "L") ? 1 : 0;
                1: load_match <= (rx_shift == "O") ? 2 : 0;
                2: load_match <= (rx_shift == "A") ? 3 : 0;
                3: begin
                    load_match <= 0;
                    if (rx_shift == "D") begin
                        load_state <= 1;
                        resident_length <= 0;
                        dot_match <= 0;
                        exec_match <= 0;
                        caps_match <= 0;
                    end
                end
                default: load_match <= 0;
            endcase
            case (exec_match)
                0: exec_match <= (rx_shift == "E") ? 1 : 0;
                1: exec_match <= (rx_shift == "X") ? 2 : 0;
                2: exec_match <= (rx_shift == "E") ? 3 : 0;
                3: begin
                    exec_match <= 0;
                    if (rx_shift == "C") exec_state <= 1;
                end
                default: exec_match <= 0;
            endcase
            case (matrix_load_match)
                0: matrix_load_match <= (rx_shift == "M") ? 1 : 0;
                1: matrix_load_match <= (rx_shift == "W") ? 2 : 0;
                2: matrix_load_match <= (rx_shift == "G") ? 3 : 0;
                3: begin
                    matrix_load_match <= 0;
                    if (rx_shift == "T") begin
                        matrix_load_state <= 1;
                        matrix_valid <= 0;
                        caps_match <= 0;
                        dot_match <= 0;
                        load_match <= 0;
                        exec_match <= 0;
                        matrix_exec_match <= 0;
                    end
                end
                default: matrix_load_match <= 0;
            endcase
            case (matrix_exec_match)
                0: matrix_exec_match <= (rx_shift == "M") ? 1 : 0;
                1: matrix_exec_match <= (rx_shift == "V") ? 2 : 0;
                2: matrix_exec_match <= (rx_shift == "E") ? 3 : 0;
                3: begin
                    matrix_exec_match <= 0;
                    if (rx_shift == "C") begin
                        matrix_exec_state <= 1;
                        caps_match <= 0;
                        dot_match <= 0;
                        load_match <= 0;
                        exec_match <= 0;
                        matrix_load_match <= 0;
                    end
                end
                default: matrix_exec_match <= 0;
            endcase
            if(io_match==0) begin
                if(rx_shift=="@") begin io_command<=32'h40; io_match<=1; end
            end else if(io_match<3) begin
                io_command<={io_command[23:0],rx_shift}; io_match<=io_match+1'b1;
            end else begin
                io_match<=0;
                if({io_command[23:0],rx_shift}=="@BUF") io_state<=1;
                else if({io_command[23:0],rx_shift}=="@BEG") checkpoint_begin_pending<=1;
                else if({io_command[23:0],rx_shift}=="@END") begin
                    response_active<=1; response_kind<=7; response_index<=0;
                end
                caps_match<=0;dot_match<=0;load_match<=0;exec_match<=0;
                matrix_load_match<=0;matrix_exec_match<=0;
            end
        end else if(rx_ready && checkpoint_begin_pending) begin
            checkpoint_begin_pending<=0;
            checkpoint_operation<=rx_shift<=2 ? rx_shift : 0;
            checkpoint_bytes<=0; checkpoint_crc<=32'hffffffff;
            response_active<=1;response_kind<=6;response_index<=0;
        end else if(rx_ready && io_state!=0 && io_state<9) begin
            case(io_state)
                1: begin io_mode<=rx_shift;io_state<=2;end
                2: begin io_length[7:0]<=rx_shift;io_state<=3;end
                3: begin
                    io_length[8]<=rx_shift[0];
                    io_index<=0;io_output_length<=0;io_input_crc<=32'hffffffff;io_output_crc<=32'hffffffff;
                    io_write_count<=0;io_crc_ready<=0;
                    io_bad<=io_mode>5 || ((io_mode==1 || io_mode==3) && io_length[1:0]!=0) ||
                        ((io_mode==2 || io_mode==4) && io_length[0]!=0);
                    if(rx_shift>1 || {rx_shift,io_length[7:0]}==0 || {rx_shift,io_length[7:0]}>256) begin
                        io_bad<=1;io_state<=9;
                    end else io_state<=4;
                end
                4: begin io_expected_crc[7:0]<=rx_shift;io_state<=5;end
                5: begin io_expected_crc[15:8]<=rx_shift;io_state<=6;end
                6: begin io_expected_crc[23:16]<=rx_shift;io_state<=7;end
                7: begin io_expected_crc[31:24]<=rx_shift;io_state<=8;end
                8: begin
                    io_input_crc<=crc_byte(io_input_crc,rx_shift);
                    if(checkpoint_operation!=0) begin
                        checkpoint_crc<=crc_byte(checkpoint_crc,rx_shift);
                        checkpoint_bytes<=checkpoint_bytes+1'b1;
                    end
                    if(io_mode==0 || io_mode==5) begin
                        io_write_word<={24'b0,rx_shift};io_write_count<=1;
                    end else if(io_mode==1 || io_mode==3) begin
                        io_word[io_index[1:0]*8 +: 8]<=rx_shift;
                        if(io_index[1:0]==3) begin
                            io_write_word<={16'b0,io_encoded};io_write_count<=2;
                        end
                    end else if(io_mode==2 || io_mode==4) begin
                        io_word[io_index[0]*8 +: 8]<=rx_shift;
                        if(io_index[0]) begin
                            io_write_word<=io_decoded;io_write_count<=4;
                        end
                    end
                    if(io_index+1'b1==io_length) begin
                        io_state<=9;io_crc_index<=0;
                        if((~crc_byte(io_input_crc,rx_shift))!=io_expected_crc || io_bad) begin
                            io_bad<=1;io_output_length<=0;io_write_count<=0;checkpoint_operation<=0;
                        end
                    end else io_index<=io_index+1'b1;
                end
            endcase
        end else if (rx_ready && dot_state == 1) begin
            if (rx_shift != 0 && rx_shift <= 64) begin
                dot_length <= rx_shift[6:0];
                dot_index <= 0;
                dot_accumulator <= 0;
                dot_state <= 2;
            end else begin
                dot_state <= 0;
            end
        end else if (rx_ready && dot_state == 2) begin
            dot_lhs <= rx_shift;
            dot_state <= 3;
        end else if (rx_ready && dot_state == 3) begin
            if (dot_index + 1'b1 == dot_length) begin
                dot_result <= dot_accumulator + dot_lhs * $signed(rx_shift);
                dot_state <= 0;
                response_active <= 1;
                response_kind <= 1;
                response_index <= 0;
            end else begin
                dot_accumulator <= dot_accumulator + dot_lhs * $signed(rx_shift);
                dot_index <= dot_index + 1'b1;
                dot_state <= 2;
            end
        end else if (rx_ready && load_state == 1) begin
            if (rx_shift != 0 && rx_shift <= 64) begin
                load_length <= rx_shift[6:0];
                load_index <= 0;
                load_state <= 2;
            end else begin
                load_state <= 0;
            end
        end else if (rx_ready && load_state == 2) begin
            resident_weights[load_index] <= rx_shift;
            if (load_index + 1'b1 == load_length) begin
                resident_length <= load_length;
                load_state <= 0;
                response_active <= 1;
                response_kind <= 2;
                response_index <= 0;
            end else begin
                load_index <= load_index + 1'b1;
            end
        end else if (rx_ready && exec_state == 1) begin
            if (rx_shift != 0 && rx_shift <= 64 && rx_shift[6:0] == resident_length) begin
                exec_length <= rx_shift[6:0];
                exec_index <= 0;
                exec_accumulator <= 0;
                exec_state <= 2;
            end else begin
                exec_state <= 0;
            end
        end else if (rx_ready && exec_state == 2) begin
            if (exec_index + 1'b1 == exec_length) begin
                dot_result <= exec_accumulator +
                    resident_weights[exec_index] * $signed(rx_shift);
                exec_state <= 0;
                response_active <= 1;
                response_kind <= 1;
                response_index <= 0;
            end else begin
                exec_accumulator <= exec_accumulator +
                    resident_weights[exec_index] * $signed(rx_shift);
                exec_index <= exec_index + 1'b1;
            end
        end else if (rx_ready && matrix_load_state == 1) begin
            if (rx_shift != 0 && rx_shift <= 8) begin
                matrix_rows <= rx_shift[3:0];
                matrix_load_state <= 2;
            end else begin
                matrix_load_state <= 0;
            end
        end else if (rx_ready && matrix_load_state == 2) begin
            if (rx_shift == 64) begin
                matrix_columns <= rx_shift[6:0];
                matrix_load_index <= 0;
                matrix_load_state <= 3;
            end else begin
                matrix_load_state <= 0;
            end
        end else if (rx_ready && matrix_load_state == 3) begin
            matrix_weights[matrix_load_index] <= rx_shift;
            if (matrix_load_index + 1'b1 == matrix_rows * matrix_columns) begin
                matrix_load_state <= 0;
                matrix_valid <= 1;
                response_active <= 1;
                response_kind <= 4;
                response_index <= 0;
            end else begin
                matrix_load_index <= matrix_load_index + 1'b1;
            end
        end else if (rx_ready && matrix_exec_state == 1) begin
            if (matrix_valid && rx_shift == matrix_columns && matrix_rows != 0) begin
                matrix_exec_index <= 0;
                for (matrix_row = 0; matrix_row < 8; matrix_row = matrix_row + 1)
                    matrix_accumulators[matrix_row] <= 0;
                matrix_exec_state <= 2;
            end else begin
                matrix_exec_state <= 0;
            end
        end else if (rx_ready && matrix_exec_state == 2) begin
            for (matrix_row = 0; matrix_row < 8; matrix_row = matrix_row + 1) begin
                if (matrix_row < matrix_rows) begin
                    if (matrix_exec_index + 1'b1 == matrix_columns)
                        matrix_results[matrix_row] <= matrix_accumulators[matrix_row] +
                            matrix_weights[matrix_row * 64 + matrix_exec_index] * $signed(rx_shift);
                    else
                        matrix_accumulators[matrix_row] <= matrix_accumulators[matrix_row] +
                            matrix_weights[matrix_row * 64 + matrix_exec_index] * $signed(rx_shift);
                end
            end
            if (matrix_exec_index + 1'b1 == matrix_columns) begin
                matrix_exec_state <= 0;
                response_active <= 1;
                response_kind <= 3;
                response_index <= 0;
            end else begin
                matrix_exec_index <= matrix_exec_index + 1'b1;
            end
        end

        // One write port and one synchronous read port infer an ECP5 EBR.
        if(io_write_count!=0) begin
            io_buffer[io_output_length]<=io_write_word[7:0];
            io_write_word<=io_write_word >> 8;
            io_write_count<=io_write_count-1'b1;
            io_output_length<=io_output_length+1'b1;
        end
        io_read_value<=io_buffer[io_read_address];
        if(io_state==9 && io_write_count==0) begin
            if(!io_crc_ready) io_crc_ready<=1;
            else if(io_crc_index<io_output_length) begin
                io_output_crc<=crc_byte(io_output_crc,io_read_value);
                io_crc_index<=io_crc_index+1'b1;
            end else begin
                io_state<=0;response_active<=1;response_kind<=5;response_index<=0;
            end
        end

        // Bound incomplete headers/payloads; a partial LOAD may have overwritten
        // resident data, so invalidate caches rather than execute a mixed matrix.
        if (rx_ready || !request_pending) request_idle <= 0;
        else if (request_idle == REQUEST_TIMEOUT_CYCLES - 1) begin
            request_idle <= 0;
            caps_match <= 0; dot_match <= 0; load_match <= 0; exec_match <= 0;
            matrix_load_match <= 0; matrix_exec_match <= 0;
            dot_state <= 0; load_state <= 0; exec_state <= 0;
            matrix_load_state <= 0; matrix_exec_state <= 0;
            resident_length <= 0; matrix_valid <= 0;
            io_match<=0;io_state<=0;io_write_count<=0;checkpoint_begin_pending<=0;checkpoint_operation<=0;
        end else request_idle <= request_idle + 1'b1;

        if (!tx_busy) begin
            tx_line <= 1;
            if (fifo_count != 0) begin
                tx_shift <= {1'b1, tx_fifo[fifo_read], 1'b0};
                tx_bit <= 0;
                tx_clock <= CLOCKS_PER_BIT - 1;
                tx_line <= 0;
                tx_busy <= 1;
            end
        end else if (tx_clock == 0) begin
            if (tx_bit == 9) begin
                tx_busy <= 0;
                tx_line <= 1;
            end else begin
                tx_bit <= tx_bit + 1'b1;
                tx_shift <= {1'b1, tx_shift[9:1]};
                tx_line <= tx_shift[1];
                tx_clock <= CLOCKS_PER_BIT - 1;
            end
        end else begin
            tx_clock <= tx_clock - 1'b1;
        end

        if (fifo_push) begin
            tx_fifo[fifo_write] <= response_byte;
            fifo_write <= fifo_write + 1'b1;
            if ((response_kind == 0 && response_index == 15) ||
                (response_kind == 1 && response_index == 7) ||
                (response_kind == 2 && response_index == 3) ||
                (response_kind == 3 && response_index + 1'b1 == 5 + matrix_rows * 4) ||
                (response_kind == 4 && response_index == 3) ||
                (response_kind == 5 && response_index == io_output_length+10) ||
                (response_kind == 6 && response_index == 3) ||
                (response_kind == 7 && response_index == 16)) begin
                response_active <= 0;
            end else begin
                response_index <= response_index + 1'b1;
            end
        end
        if (fifo_pop) fifo_read <= fifo_read + 1'b1;
        case ({fifo_push, fifo_pop})
            2'b10: fifo_count <= fifo_count + 1'b1;
            2'b01: fifo_count <= fifo_count - 1'b1;
            default: fifo_count <= fifo_count;
        endcase
    end
endmodule

// A bounded session lease prevents a crashed/disconnected host from leaving
// a misleading activity indicator. Heartbeats do not restart the double flash.
module mimir_status_led #(
    parameter integer FAST_CYCLES = 2_083_333,
    parameter integer SLOW_CYCLES = 12_500_000,
    parameter integer LEASE_CYCLES = 125_000_000
) (
    input wire clk, input wire start, input wire keepalive, input wire stop,
    output reg on = 0, output wire active
);
    reg [$clog2(LEASE_CYCLES+1)-1:0] lease = 0;
    reg [$clog2(SLOW_CYCLES+FAST_CYCLES+1)-1:0] ticks = 0;
    reg [1:0] phase = 0;
    reg initializing = 0;
    assign active = lease != 0;
    always @(posedge clk) begin
        if (stop) begin
            lease <= 0; ticks <= 0; initializing <= 0; on <= 0;
        end else if (start) begin
            lease <= LEASE_CYCLES; ticks <= 0; phase <= 0;
            initializing <= 1; on <= 1;
        end else if (lease == 0 && keepalive) begin
            // Computation may resume after an idle interval without a handshake.
            lease <= LEASE_CYCLES; ticks <= 0; on <= 0;
        end else if (lease != 0) begin
            if (keepalive) lease <= LEASE_CYCLES;
            else lease <= lease - 1'b1;
            if (lease == 1 && !keepalive) begin
                on <= 0; initializing <= 0; ticks <= 0;
            end else if (initializing) begin
                if (ticks == FAST_CYCLES - 1) begin
                    ticks <= 0;
                    if (phase == 3) begin initializing <= 0; on <= 0; end
                    else begin phase <= phase + 1'b1; on <= phase == 1; end
                end else ticks <= ticks + 1'b1;
            end else if (ticks == SLOW_CYCLES - 1) begin
                ticks <= 0; on <= !on;
            end else ticks <= ticks + 1'b1;
        end
    end
endmodule
