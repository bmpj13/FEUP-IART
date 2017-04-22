from collections import namedtuple
from keyword import iskeyword
import re

def NotDone(msg):
    raise NotImplemented(msg)

def nominal(spec):
    """
    Create an ARFF nominal (enumerated) data type
    """
    spec = spec.lstrip("{ \t").rstrip("} \t")
    good_values = set(val.strip() for val in spec.split(","))

    def fn(s):
        s = s.strip()
        if s in good_values:
            return s
        else:
            raise ValueError("'{}' is not a recognized value".format(s))

    # patch docstring
    fn.__name__ = "nominal"
    fn.__doc__ = """
    ARFF nominal (enumerated) data type

    Legal values are {}
    """.format(sorted(good_values))
    return fn

def numeric(s):
    """
    Convert string to int or float
    """
    try:
        return int(s)
    except ValueError:
        return float(s)

field_maker = {
    "date":       (lambda spec: NotDone("date data type not implemented")),
    "integer":    (lambda spec: int),
    "nominal":    (lambda spec: nominal(spec)),
    "numeric":    (lambda spec: numeric),
    "string":     (lambda spec: str),
    "real":       (lambda spec: float),
    "relational": (lambda spec: NotDone("relational data type not implemented")),
}

def file_lines(fname):
    # lazy file reader; ensures file is closed when done,
    # returns lines without trailing spaces or newline
    with open(fname) as inf:
        for line in inf:
            yield line.rstrip()

def no_data_yet(*items):
    raise ValueError("AarfRow not fully defined (haven't seen a @data directive yet)")

def make_field_name(s):
    """
    Mangle string to make it a valid Python identifier
    """
    s = s.lower()                               # force to lowercase
    s = "_".join(re.findall("[a-z0-9]+", s))    # strip all invalid chars; join what's left with "_"
    if iskeyword(s) or re.match("[0-9]", s):    # if the result is a keyword or starts with a digit
        s = "f_"+s                              #   make it a safe field name
    return s

class ArffReader:
    line_types = ["blank", "comment", "relation", "attribute", "data"]

    def __init__(self, fname):
        # get input file
        self.fname = fname
        self.lines = file_lines(fname)

        # prepare to read file header
        self.relation = '(not specified)'
        self.data_names = []
        self.data_types = []
        self.dtype = no_data_yet

        # read file header
        line_tests = [
            (getattr(self, "line_is_{}".format(item)), getattr(self, "line_do_{}".format(item)))
            for item in self.__class__.line_types
        ]
        for line in self.lines:
            for is_, do in line_tests:
                if is_(line):
                    done = do(line)
                    break
            if done:
                break

        # use header fields to build data type (and make it print as requested)
        class ArffRow(namedtuple('ArffRow', self.data_names)):
            __slots__ = ()
            def __str__(self):
                items = (getattr(self, field) for field in self._fields)
                return "({})".format(", ".join(repr(it) for it in items))
        self.dtype = ArffRow

    #
    # figure out input-line type
    #

    def line_is_blank(self, line):
        return not line

    def line_is_comment(self, line):
        return line.lower().startswith('%')

    def line_is_relation(self, line):
        return line.lower().startswith('@relation')

    def line_is_attribute(self, line):
        return line.lower().startswith('@attribute')

    def line_is_data(self, line):
        return line.lower().startswith('@data')

    #
    # handle input-line type
    #

    def line_do_blank(self, line):
        pass

    def line_do_comment(self, line):
        pass

    def line_do_relation(self, line):
        self.relation = line[10:].strip()

    def line_do_attribute(self, line):
        m = re.match(
            "^@attribute"           #   line starts with '@attribute'
            "\s+"                   #
            "("                     # name is one of:
                "(?:'[^']+')"       #   ' string in single-quotes '
                "|(?:\"[^\"]+\")"   #   " string in double-quotes "
                "|(?:[^ \t'\"]+)"   #   single_word_string (no spaces)
            ")"                     #
            "\s+"                   #
            "("                     # type is one of:
                "(?:{[^}]+})"       #   { set, of, nominal, values }
                "|(?:\w+)"          #   datatype
            ")"                     #
            "\s*"                   #
            "("                     # spec string
                ".*"                #   anything to end of line
            ")$",                   #
            line, flags=re.I)       #   case-insensitive
        if m:
            name, type_, spec = m.groups()
            self.data_names.append(make_field_name(name))
            if type_[0] == '{':
                type_, spec = 'nominal', type_
            self.data_types.append(field_maker[type_](spec))
        else:
            raise ValueError("failed parsing attribute line '{}'".format(line))

    def line_do_data(self, line):
        return True  # flag end of header

    #
    # make the class iterable
    #

    def __iter__(self):
        return self

    def next(self):
        """
        Return one data row at a time
        """
        data = next(self.lines).split(',')
        return self.dtype(*(fn(dat) for fn,dat in zip(self.data_types, data)))


# http://stackoverflow.com/questions/36106712/how-can-i-limit-iterations-of-a-loop-in-python
# http://stackoverflow.com/questions/22187589/how-can-i-parse-an-arff-file-without-using-external-libraries-in-python
'''
for row in islice(ArffReader('data/dataset.arff'), start, end):
    print(row)
'''

# limit for search
from itertools import islice
start = 0
end = 0

def next_training_set(batch_size):
    global start
    global end
    end = start + batch_size
    rows = islice(ArffReader('data/dataset.arff'), start, end)
    start += batch_size
    return get_sets(rows)


def test_set(train_set_size):
    test_start = train_set_size + 1
    rows = islice(ArffReader('data/dataset.arff'), test_start)
    return get_sets(rows)

def reset_batch_start():
    global start
    start = 0


def get_sets(rows):
    epoch_x = []
    epoch_y = []
    for row in rows:
        epoch_x.append(row[:-1])
        epoch_y.append(row[-1])
    return epoch_x, epoch_y
