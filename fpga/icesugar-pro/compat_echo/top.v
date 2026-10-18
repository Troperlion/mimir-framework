module top (
    input wire clk25,
    input wire serial_rx,
    output wire serial_tx
);
    localparam integer CLOCK_HZ = 25_000_000;
    localparam integer BAUD = 115_200;
    localparam integer CLOCKS_PER_BIT = CLOCK_HZ / BAUD;

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

    localparam integer FIFO_DEPTH = 16;
    reg [7:0] tx_fifo [0:FIFO_DEPTH - 1];
    reg [3:0] fifo_write = 0;
    reg [3:0] fifo_read = 0;
    reg [4:0] fifo_count = 0;

    wire fifo_pop = !tx_busy && fifo_count != 0;
    wire fifo_push = rx_ready && (fifo_count != FIFO_DEPTH || fifo_pop);

    assign serial_tx = tx_line;

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
            rx_clock <= rx_clock - 1;
        end

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
                tx_bit <= tx_bit + 1;
                tx_shift <= {1'b1, tx_shift[9:1]};
                tx_line <= tx_shift[1];
                tx_clock <= CLOCKS_PER_BIT - 1;
            end
        end else begin
            tx_clock <= tx_clock - 1;
        end

        if (fifo_push) begin
            tx_fifo[fifo_write] <= rx_shift;
            fifo_write <= fifo_write + 1'b1;
        end
        if (fifo_pop) begin
            fifo_read <= fifo_read + 1'b1;
        end
        case ({fifo_push, fifo_pop})
            2'b10: fifo_count <= fifo_count + 1'b1;
            2'b01: fifo_count <= fifo_count - 1'b1;
            default: fifo_count <= fifo_count;
        endcase
    end
endmodule