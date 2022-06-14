KEYBOARD_MAP = [
    #       '   +   ,   -   .   0   1   2   3   4   5   6   7   8   9   <   `   a   b   c   d   e   f   g   h   i   j   k   l   m   n   o   p   q   r   s   t   u   v   w   x   y   z   ¡   ´   º   ç   ñ
    [0, 1,  2,  1,  5,  3,  4,  3,  12, 11, 10, 9,  8,  7,  6,  5,  4,  13, 2,  12, 8,  10, 10, 10, 9,  8,  7,  5,  6,  5,  4,  6,  7,  4,  3,  12, 9,  11, 10, 8,  9,  11, 11, 9,  12, 1,  2,  13, 2,  3   ], # 1
    [-1, 0, 3,  1,  5,  3,  4,  4,  13, 12, 11, 10, 9,  8,  7,  6,  5,  13, 2,  12, 8,  10, 10, 10, 9,  8,  7,  6,  6,  5,  4,  6,  7,  5,  4,  12, 9,  11, 8,  6,  9,  11, 11, 7,  12, 2,  2,  11, 1,  3   ], # 2
    [-1,-1, 0,  2,  3,  3,  3,  1,  10, 9,  8,  7,  6,  5,  4,  3,  2,  11, 1,  10, 6,  8,  8,  8,  7,  6,  5,  3,  4,  3,  2,  4,  5,  2,  1,  10, 7,  9,  6,  4,  7,  9,  9,  5,  10, 1,  2,  11, 3,  2   ], # 3
    [-1,-1, -1, 0,  4,  2,  3,  3,  12, 11, 10, 9,  8,  7,  6,  5,  4,  12, 1,  11, 9,  9,  9,  9,  8,  7,  6,  4,  5,  4,  3,  5,  6,  3,  2,  11, 8,  10, 7,  5,  8,  10, 10, 6,  11, 1,  1,  13, 1,  2   ], # 4
    [-1,-1, -1, -1, 0,  2,  1,  3,  10, 9,  8,  7,  6,  5,  4,  3,  3,  8,  3,  8,  3,  5,  6,  7,  5,  4,  3,  2,  2,  1,  1,  1,  2,  2,  2,  9,  6,  7,  5,  3,  4,  8,  6,  4,  7,  4,  3,  11, 4,  2   ], # 5
    [-1,-1, -1, -1, -1, 0,  1,  3,  12, 11, 10, 9,  8,  7,  6,  5,  4,  10, 2,  10, 7,  7,  8,  9,  7,  6,  5,  4,  4,  3,  2,  3,  4,  3,  2,  11, 8,  9,  7,  5,  6,  10, 8,  6,  9,  3,  1,  13, 2   1   ], # 6
    [-1,-1, -1, -1, -1, -1, 0,  3,  11, 10, 9,  8,  7,  6,  5,  4,  3,  9,  2,  9,  4,  6,  7,  8,  6,  5,  4,  3,  3,  2,  1,  2,  3,  2,  2,  10, 7,  8,  6,  4,  5,  9,  7,  5,  8,  3,  3,  12, 3,  1   ], # 7
    [-1,-1, -1, -1, -1, -1, -1, 0,  9,  8,  7,  6,  5,  4,  3,  2,  1,  10, 2,  9,  5,  7,  7,  7,  6,  5,  4,  2,  2,  2,  2,  3,  3,  1,  1,  10, 6,  8,  5,  3,  6,  8,  8,  4,  9,  2,  3,  10, 4,  2   ], # 8
    [-1,-1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  3,  11, 2,  7,  5,  4,  3,  5,  6,  7,  8,  8,  9,  10, 9,  8,  9,  10, 1,  4,  3,  5,  7,  6,  2,  4,  6,  3,  11, 12, 1,  13, 11  ], # 9
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  3,  10, 2,  6,  4,  3,  2,  4,  5,  6,  7,  7,  8,  9,  8,  7,  8,  9,  1,  3,  2,  4,  6,  5,  1,  3,  5,  3,  10, 11, 2,  12, 10  ], # 10
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  3,  9,  2,  5,  3,  1,  3,  4,  5,  6,  6,  6,  7,  8,  7,  6,  7,  8,  2,  2,  2,  3,  5,  4,  1,  3,  4,  3,  9,  10, 3,  11, 9   ], # 11
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  4,  8,  3,  3,  2,  1,  2,  3,  3,  4,  5,  5,  6,  7,  6,  5,  6,  7,  3,  1,  2,  2,  5,  3,  1,  3,  3,  3,  8,  9,  4,  10, 8   ], # 12
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  7,  4,  3,  3,  2,  2,  2,  2,  3,  4,  4,  5,  6,  5,  4,  5,  6,  4,  1,  3,  1,  3,  3,  3,  3,  2,  4,  7,  8,  5,  9,  7   ], # 13
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0   1,  2,  3,  6,  6,  5,  3,  3,  3,  3,  2,  2,  2,  3,  3,  4,  5,  4,  3,  4,  5,  5,  2,  4,  1,  2,  3,  4,  1,  1,  5,  6,  7,  6,  8,  6   ], # 14
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  7,  5,  6,  3,  4,  4,  4,  3,  2,  2,  3,  2,  1   4,  3,  3,  3,  4,  6,  3,  5,  2,  1,  3,  5,  5,  1,  6,  5,  6,  7,  7,  6   ], # 15
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  8,  6,  5,  3,  5,  6,  5,  4,  3,  2,  1,  2,  2,  3,  3,  3,  2,  3,  7,  4,  6,  3,  1,  4,  6,  6,  2,  6,  4,  5,  8,  6,  4   ], # 16
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  9,  3,  6,  4,  6,  6,  6,  5,  4,  3,  1,  2,  2,  3,  3,  3,  1,  2,  8,  5,  7,  4,  5,  5,  5,  7,  3,  8,  3,  4,  9,  3,  3   ], # 17
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  11, 1,  5,  3,  3,  3,  4,  5,  6,  8,  8,  8,  9,  7,  6,  9,  10, 2,  4,  2,  5,  7,  4,  3,  2,  6,  2,  13, 11, 3,  12, 10  ], # 18
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  10, 6,  8,  8,  8,  7,  6,  5,  3,  4,  3,  2,  4,  5,  2,  1,  10, 7,  9,  6,  4,  7,  9,  9,  5,  8,  1,  1,  12, 2,  1   ], # 19
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  5,  3,  2,  2,  3,  4,  5,  7,  6,  7,  8,  7,  6,  8,  9,  1,  3,  1,  4,  6,  4,  1,  2,  5,  1,  10, 10, 3,  11, 9   ], # 20
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  2,  3,  4,  2,  1,  1,  3,  2,  3,  4,  2,  1,  4,  5,  6,  3,  4,  2,  2,  1,  4,  3,  2,  4,  7,  6,  8,  7,  5   ], # 21
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  1,  2,  3,  5,  4,  5,  6,  4,  3,  6,  7,  4,  2,  2,  2,  4,  1,  3,  1,  3,  2,  9,  8,  6,  9,  7   ], # 22
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  1,  2,  3,  5,  4,  5,  6,  5,  4,  6,  7,  3,  1,  1,  2,  4,  2,  2,  1,  3,  2,  9,  8,  5,  9,  7   ], # 23
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  2,  3,  4,  5,  5,  6,  7,  7,  6,  6,  7,  2,  1,  1,  2,  4,  3,  1,  2,  3,  2,  9,  9,  4,  10, 8   ], # 24
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 25
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 26
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 27
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 30
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 31
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 32
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 33
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 34
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 35
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 36
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 37
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 38
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 39
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 40
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 41
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 42
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 43
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 44
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 45
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 46
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  ], # 47
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  14, 12  ], # 48
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  2   ], # 49
    [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0   ], # 50
]