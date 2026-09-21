# This test is written in assembly to ensure consistent 
# instruciton ordering with rave markers
# This matters for this one, since we need to count 
# the number of instructions in each region, which 
# includes the overlapping markers themselves

# RAVE syntax:
# {-1} is any register containing -1
# -1 is used to indicate that the string should be read until \0
# rave_begin_region:
#   la <reg>, &str
#   add zero, <reg>, {-1}
#
# rave_end_region:
#   sub zero, <reg>, {-1}

.section .rodata
region_1:
  .asciz "region_1"

region_2:
  .asciz "region_2"

.section .text
.globl _start

_start:
  xori a6, zero, 1

  li a1, -1
# begin("region_1")
  la a0, region_1
  add zero, a0, a1

  xori a6, zero, 1
  xori a6, zero, 1

# begin("region_2")
  la a2, region_2
  add zero, a2, a1

  xori a6, zero, 1
  xori a6, zero, 1

# end("region_1")
  sub zero, a0, a1

  xori a6, zero, 1

# end("region_2")
  sub zero, a2, a1

  xori a6, zero, 1

end:
    li a0, 0
    li a7, 93 # syscall number for exit
    ecall
