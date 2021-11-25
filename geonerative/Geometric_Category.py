from geonerative.token_names import *

class Geo:
    def __init__(self, type_, id_, properties):
        self.type = type_
        self.id = id_
        self.properties = properties
        if 'visibility' not in self.properties:
            self.properties['visibility'] = True
        if self.type == 'rectangle':
            if 'mode' not in self.properties:
                self.properties['mode']='CENTER'

    def __repr__(self):
        if not self.type:
            return f"noType #{self.id}"
        result = f"{self.type} #{self.id}"
        for prop, value in self.properties.items():
            if not prop in ('type', 'id_'):
                result += f" : {prop} = {value}"
        return result

    def translate(self, x, y):
        self.x_loc += x
        self.y_loc += y
        for obj in self.list:
            obj.translate(x, y)

    def set_x(self, x):
        self.x_loc = x

    def set_y(self, y):
        self.y_loc = y

    def set_coordinates(self, x, y):
        self.x_loc = x
        self.y_loc = y

    def set_radius(self, r):
        self.radius = r

    def show(self):
        self.properties['visibility'] = True

    def hide(self):
        self.properties['visibility'] = False

    def modify(self, properties):
        self.properties.update(properties)

    def draw(self, canvas):
        if not self.properties['visibility']:
            return
        properties=self.properties
        match self.type:
            case 'circle':
                canvas.create_circle(properties['x_loc'], properties['y_loc'],
                                     properties['radius'],
                                     fill=properties['color'])
            case 'rectangle':
                match properties['mode']:
                    case 'CENTER':
                        x_0, y_0=properties['x_loc']-properties['width']/2, properties['y_loc']-properties['height']/2
                        x_1, y_1=properties['x_loc']+properties['width']/2, properties['y_loc']+properties['height']/2
                    case 'CORNER':
                        x_0,y_0=properties['x_loc'], properties['y_loc']
                        x_1,y_1=properties['x_loc']-properties['width'], properties['y_loc']-properties['height']
                canvas.create_rectangle(x_0,y_0,x_1,y_1, fill=properties['color'])