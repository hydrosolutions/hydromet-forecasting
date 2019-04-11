

def to_str(str_or_unicode):
    if isinstance(str_or_unicode, unicode):
        return str_or_unicode.encode('utf-8')

    return str_or_unicode
