class Format:

    @staticmethod
    def format(variable, key):
        if type(key) == type(dict()):
            return key[variable]
        if key == "int":
            return int(variable)
        if key == "float":
            return round(float(variable), 1)
        if key == "month":
            if variable == "Jan" or "January":
                return 0
            if variable == "Feb" or "February":
                return 1
            if variable == "Mar" or "March":
                return 2
            if variable == "Apr" or "April":
                return 3
            if variable == "May":
                return 4
            if variable == "Jun" or "June":
                return 5
            if variable == "Jul" or "July":
                return 6
            if variable == "Aug" or "August":
                return 7
            if variable == "Sep" or "September":
                return 8
            if variable == "Oct" or "October":
                return 9
            if variable == "Nov" or "November":
                return 10
            if variable == "Dec" or "December":
                return 11
        if key == "boolean":
            variable.upper()
            if variable == "TRUE":
                return 1
            variable.upper()
            if variable == "FALSE":
                return 0
        else:
            print(variable)
