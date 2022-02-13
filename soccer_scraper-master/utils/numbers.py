def percent(number, places=2, string_percent = False):
    number = round(number, places)*100
    if string_percent:
        number = str(number) + '%'
    return number

