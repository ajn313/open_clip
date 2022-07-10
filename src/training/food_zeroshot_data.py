import ast

with open( r'./metadata/food.txt', 'r' ) as file:
    food_classnames = ast.literal_eval(file.read())

food_template = [
    lambda c: f'a photo of the {c}, a type of food.',
    lambda c: f'a close-up photo of the {c}, a type of food.',
    lambda c: f'a rendition of the {c}, a type of food.',
    lambda c: f'a photo of a large {c}, a type of food.',
    lambda c: f'itap of a {c}, a type of food.',
]