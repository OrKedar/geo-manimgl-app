global_objects = {}


class Geo:
    def __init__(self, type_, id_, properties):
        self.type = type_
        if type_ == 'circle':
            self.properties = properties
            self.x_loc, self.y_loc = properties['x_loc'], properties['y_loc']
            self.radius = properties['radius']
            self.color = properties.get('color', '#BBB')
            self.id = id_
            self.visibility = properties.get('visibility', True)

    def __repr__(self):
        return f'{self.properties}'

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
        self.visibility = True

    def hide(self):
        self.visibility = False

    def draw(self, canvas):
        if self.visibility:
            canvas.create_circle(self.x_loc, self.y_loc, self.radius, fill=self.color)
