db.define_table('image',
    Field('title', required=True, notnull=True, unique=True),
    Field('image', 'upload'),
    Field('createat', 'datetime', default=request.now),
    format = '%(title)s'
)
