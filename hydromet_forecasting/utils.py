import os
import gettext


def to_str(str_or_unicode):
    if isinstance(str_or_unicode, unicode):
        return str_or_unicode.encode('utf-8')

    return str_or_unicode


def activate(language):
    locales_path = os.path.join(os.path.dirname(__file__), 'locales')
    locales = os.environ.get('LOCALES_PATH', locales_path)
    t = gettext.translation('messages', locales, languages=[language])
    t.install()
