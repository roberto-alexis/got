db.define_table('image',
    Field('title'),
    Field('image', 'upload'),
    format = '%(title)s')
