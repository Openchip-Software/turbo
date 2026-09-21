# This test is written in assembly to ensure consistent 
# instruciton ordering with rave markers
# This matters for this one, since we need to count 
# the number of instructions in each region, which 
# includes the nested markers themselves

# RAVE syntax:
# TODO

.section .text
.globl _start

_start:

rave_setup:
  # rave_name_event(1000, "abc");
  li a1, -1
  li a0, 1000
  and zero, a0, a1

  # abc = 0x61 0x62 0x63
  # 16 26 36
  li zero, -1

  lui zero, 1
  lui zero, 6
  
  lui zero, 2
  lui zero, 6

  lui zero, 3
  lui zero, 6

  li zero,-1

  # rave_name_value(1000, 0, "end")
  and zero, a0, zero
  # end = 0x65 0x6e 0x64
  # 56 e6 46
  li zero, -1

  lui zero, 5
  lui zero, 6

  lui zero, 2
  lui zero, 6

  lui zero, 3
  lui zero, 6

  li zero, -1

  # rave_name_value(1000, 1, "region_1")
  li a2, 1
  and zero, a0, a2
  # region_1 = ['0x72', '0x65', '0x67', '0x69', '0x6f', '0x6e', '0x5f', '0x31']
  # 27 56 76 96 f6 e6 f5 13
  li zero, -1

  lui zero, 2
  lui zero, 7

  lui zero, 5
  lui zero, 6

  lui zero, 7
  lui zero, 6

  lui zero, 9
  lui zero, 6

  lui zero, 0xf
  lui zero, 6

  lui zero, 0xe
  lui zero, 6

  lui zero, 0xf
  lui zero, 5

  lui zero, 1
  lui zero, 3

  li zero, -1

  # rave_restart_trace()
  li zero, -2

main:
  xori a6, a0, 1

  # rave_event_and_value(1000, 1)
  or zero, a0, a2

  xori a6, a0, 1

  # rave_event_and_value(1000, 0)
  or zero, a0, zero

  xori a6, a0, 1


end:
    li a0, 0
    li a7, 93 # syscall number for exit
    ecall
