import sys

macheps = 1.0
while(1 + macheps != 1):
    macheps/=2
macheps*=2
print(macheps)
print(type(macheps))
print(sys.float_info.epsilon)