class Normalize():
    def normalize(input, min=sys.maxsize, max=(1-sys.maxsize)):
        if isinstance(input, (int, float)):
            input = (input - min) / (max - min)
        else:
            for x in input:
                if input[x] < min:
                    min = input[x]

                if input[x] > max:
                    max = input[x]

            for x in input:
                try:
                    input[x] = (input[x] - min) / (max - min)
                except:
                    input[x] = 0
            con = []
            for x in input:
                if input[x] != 0:
                    con.append(input[x])
                else:
                    input[x] = -0.1
            input = con
