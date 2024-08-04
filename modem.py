#%%
import numpy as np
import matplotlib.pyplot as plt

class qamModem:
    def __init__(self, modulationOrder):
        self._checkIf_M_Valid(modulationOrder)
        self.M = modulationOrder
        self.m = int(np.log2(self.M))

        self._bitToIntKernel = 1<<(np.arange(self.m - 1 ,-1,-1))

        self.constellation, self.bitList = generateQAMconstellationAndBitList(self.M)
        self._generateLLRTable()

    #----> methods
    def modulateBits(self, bits):
        idxList = self._bitToIntKernel @ bits.reshape(self.m, len(bits)//self.m, order = 'F')
        return self.constellation[idxList]

    def calculateLLR(self, x):
        x_InPhase    = x.real[np.newaxis, :, np.newaxis]
        x_Quadrature = x.imag[np.newaxis, :, np.newaxis]

        distance0 = np.min((x_InPhase - self.llrTable0)**2, axis = 2)
        distance1 = np.min((x_InPhase - self.llrTable1)**2, axis = 2)
        LLR_InPhase = (distance1 - distance0)

        distance0 = np.min((x_Quadrature - self.llrTable0)**2, axis = 2)
        distance1 = np.min((x_Quadrature - self.llrTable1)**2, axis = 2)
        LLR_Quadrature = (distance1 - distance0)

        LLRs =  np.vstack((LLR_InPhase, LLR_Quadrature)).reshape(-1, order = "F")
        return LLRs


    def plotConstellation(self):
        plt.figure(1)
        plt.plot(self.constellation.real, self.constellation.imag, ls = 'None' , marker = 'x')
        for ii in range(self.M):
            plt.text(self.constellation.real[ii], self.constellation.imag[ii], str(self.bitList[:,ii]))

    #---> Private
    def _generateLLRTable(self):
            M_sqrt = int(np.sqrt(self.M))
            ## evaluate LLR for PAM signals (considersing as if two independent PAM signals )

            constellationPAM = self.constellation[:M_sqrt].imag
            bitListPAM       = self.bitList[self.m//2:, :M_sqrt]

            self.llrTable0 =  np.zeros((self.m//2, 1, M_sqrt//2), dtype = np.float32)
            self.llrTable1 =  np.zeros((self.m//2, 1, M_sqrt//2), dtype = np.float32)

            for ii in range(self.m//2):
                idxList_0 = bitListPAM[ii, :] == 0
                idxList_1 = bitListPAM[ii, :] == 1

                self.llrTable0[ii, 0, :] = constellationPAM[idxList_0]
                self.llrTable1[ii, 0, :] = constellationPAM[idxList_1]

    def _checkIf_M_Valid(self, M):
        m = int(np.log2(M))
        flagCheck = True
        if int(2**m) != M or (m/2-m//2) != 0:
            raise Exception("M must be a power of 4 !!!")

#-----------------------------------------------------------
def _grayCoder(bit_array):
    bit_array_gray      = np.zeros_like(bit_array)
    bit_array_gray[0]  = bit_array[0]
    bit_array_gray[1:] = bit_array[:-1] ^ bit_array[1:]
    return bit_array_gray

def _grayDecoder(bit_array_gray):
    N              = len(bit_array_gray)
    bit_array      = np.zeros_like(bit_array_gray)

    bit_array[0]  = bit_array_gray[0]
    for ii in range(1, N):
        bit_array[ii] = bit_array[ii - 1] ^ bit_array_gray[ii]
    return bit_array

def _intToBinaryBitArray(n, bit_length):

    # Create a NumPy array of zeros with the specified bit length
    binary_array = np.zeros(bit_length, dtype = int)

    # Convert integer to binary using bitwise operations
    for i in range(bit_length):
        binary_array[i] = n & 1
        n >>= 1

    return np.flip(binary_array)

def _generatePAMconstellationAndBitList(M):
        m = int(np.log2(M))
        if np.mod(m, 1) != 0:
            raise ValueError('Input M must be appropriate for QAM modulation')
        #-----------------
        bitToIntKernel = 1<<(np.arange(m - 1 ,-1,-1))
        bitArrayGray   = np.zeros( (m, M), dtype = np.int32)
        constellation  = np.zeros(M, dtype = np.complex64)
        for ii in range(M):
            bitArrayGray[:, ii] = _grayCoder(_intToBinaryBitArray(ii, m))
            constellation[ii]   = ii + 1j*0

        # Reorder the bit list and constellation points w.r.t. natural order
        decimalGray = bitToIntKernel @ bitArrayGray
        sortedIdxList = np.argsort(decimalGray)
        bitArray      = bitArrayGray[:, sortedIdxList]
        constellation = constellation[sortedIdxList]
        return constellation, bitArray

def generateQAMconstellationAndBitList(M):
        m = int(np.log2(M))
        if np.mod(m/2, 1) != 0:
            raise ValueError('Input M must be appropriate for QAM modulation')
        #-----------------
        constellationPAM, bitListPAM = _generatePAMconstellationAndBitList(1<<(m//2))

        bitList = np.zeros((m, M), dtype = np.int32)
        constellation = np.zeros(M, dtype = np.complex64)
        kk = 0
        for ii in range(1<<(m//2)):
            for jj in range(1<<(m//2)):
                constellation[kk]  = constellationPAM[ii] + 1j*constellationPAM[jj]
                bitList[:m//2, kk] = bitListPAM[:, ii]
                bitList[m//2:, kk] = bitListPAM[:, jj]
                kk += 1

        constellation -= np.mean(constellation)
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation, bitList
