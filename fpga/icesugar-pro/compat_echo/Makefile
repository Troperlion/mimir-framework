TARGET := compat_echo

.PHONY: all clean

all: $(TARGET).bit

$(TARGET).json: top.v
	yosys -p "synth_ecp5 -top top -json $@" $<

$(TARGET).config: $(TARGET).json icesugar_pro.lpf
	nextpnr-ecp5 --25k --package CABGA256 --speed 6 \
		--json $< --lpf icesugar_pro.lpf --textcfg $@ --freq 25

$(TARGET).bit: $(TARGET).config
	ecppack --svf $(TARGET).svf $< $@

clean:
	rm -f $(TARGET).json $(TARGET).config $(TARGET).bit $(TARGET).svf