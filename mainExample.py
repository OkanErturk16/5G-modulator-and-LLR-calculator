#%%
import numpy as np
import modem

#%%

# 64-QAM
M = 64

# generate 64-QAM modem
myModem = modem.qamModem(M)

# number of symbols
nSymbol = int(100)

#-> this method can be used for m
#myModem.plotConstellation()

# generate bits to be transmitted
bits = np.random.randint( 0 , high = 2, size = (myModem.m*nSymbol), dtype = np.int32)

# modulate bits in accordance with 5G standard
x = myModem.modulateBits(bits)

# calculate LLRS
LLRs = myModem.calculateLLR(x)

# Hard decoding
bitsHat = np.array(LLRs < 0, dtype = np.int32)

# calculate number of errors
nError = np.sum( bitsHat != bits)
print("Num error is:", nError)

















# %%
