import json
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.addnodes import productionlist, production


def init_peg_fragments(app):
    with Path(app.srcdir, app.config.peg_data_file).open() as file:
        peg_data = json.load(file)

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
            rule_name = self.arguments[0]
            try:
                rule_data = peg_data[rule_name]
            except KeyError:
                raise LookupError(f'grammar rule {rule_name} not found in {app.config.peg_data_file}')
            productions = []
            for line in rule_data:
                productions.append(production(
                    '',
                    '',
                    nodes.Text(line),
                    tokenname=rule_name,
                    ids=[f'peg-python-grammar-{rule_name}'],
                ))
            return [
                productionlist(
                    '',
                    *productions,
                )
            ]

            # return [
            #     productionlist(
            #         '',
            #         production(
            #             '',
            #             '',
            #             nodes.Text(' (NEWLINE | '),
            #             nodes.reference(
            #                 'statement',
            #                 '',
            #                 nodes.literal(
            #                     'statement',
            #                     'statement',
            #                     classes=['xref'],
            #                 ),
            #                 refuri='compound_stmts#grammar-token-python-grammar-statement',  # XXX: generate a proper reference
            #                 internal=True,
            #             ),
            #             nodes.Text(')*'),
            #             tokenname=rule_name,
            #             ids=[f'peg-python-grammar-{rule_name}'],
            #         )
            #     )
            # ]

    app.add_directive('peg-fragment', PEGFragmentDirective)

def setup(app):
    app.add_config_value('peg_data_file', '', True)
    app.connect('builder-inited', init_peg_fragments)
    return {'version': '1.0', 'parallel_read_safe': True}
