from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.screenmanager import ScreenManager
from kivy.lang import Builder
from kivy.utils import get_color_from_hex
from kivymd.uix.relativelayout import MDRelativeLayout
from kivy.properties import StringProperty
from kivymd.uix.card import MDCard

kv = Builder.load_file("gui_style.kv")

class SearchScreen(Screen):
    pass

# search screen widgets
class SearchBarWidget(MDRelativeLayout):
    text = StringProperty()
    hint_text = StringProperty()
    
class ResultsScreen(Screen):
    def navigate_to_search_screen(self):
        self.manager.current = 'search_screen'

    def navigate_to_product_info_screen(self):
        self.manager.current = 'product_info_screen'

class ProductPreviewWidget(MDCard):
    name = StringProperty()
    details = StringProperty()
    price = StringProperty()
    image_url = StringProperty()

class ProductInfoScreen(Screen):
    def navigate_to_results_screen(self):
        self.manager.current = 'results_screen'

class InterfaceScreenManager(ScreenManager):
    pass


class WSCARApp(MDApp):

    def build(self):
        return InterfaceScreenManager()

if __name__ == "__main__":
    WSCARApp().run()