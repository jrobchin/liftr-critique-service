from kivy import properties
from kivy.uix import label

class SessionKeyLabel(label.Label):
    s_key = properties.StringProperty()