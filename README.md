This project is generated for fast & easy modulation as well as LLR calculation for 5G-type QAM signalling.
`modem` class is firstly generated and it can be used for modulating bit stream. After, the same object can be used for LLR calculation without any adjustment on the class.
It uses Numpy arrays for fast matrix manipulations, and I/Q stream is divided and LLR calculations are performed for each branch. Therefore, it is expected that the LLR calculations are performed in a quite fast manner.
