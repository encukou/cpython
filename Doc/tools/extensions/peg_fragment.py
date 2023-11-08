from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.addnodes import productionlist, production

class PEGFragmentDirective(Directive):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        #'class': directives.class_option,
        #'name': directives.unchanged,
    }

    def run(self):
        return [
            productionlist(
                '',
                production(
                    '',
                    '',
                    nodes.Text(' (NEWLINE | '),
                    nodes.reference(
                        'statement',
                        '',
                        nodes.literal(
                            'statement',
                            'statement',
                            classes=['xref'],
                        ),
                        refuri='compound_stmts#grammar-token-python-grammar-statement',  # XXX: generate a proper reference
                        internal=True,
                    ),
                    nodes.Text(')*'),
                    tokenname='file_input',
                    ids=['peg-python-grammar-file_input'],
                )
            )
        ]

def setup(app):
    app.add_directive('peg-fragment', PEGFragmentDirective)
    return {'version': '1.0', 'parallel_read_safe': True}
