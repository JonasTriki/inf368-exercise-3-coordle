# inf368-exercise-3-coordle
INF368 Spring 2020 Exercise 3 - Coordle pip package

# How to install:
First install

```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
```

then 
```
pip install git@github.com:JonasTriki/inf368-exercise-3-coordle.git
```

Possible imports
```python
from coordle.backend import Index, QueryAppenderIndex, CordDoc, RecursiveDescentParser
from coordle.preprocessing import CORD19Data
from coordle.utils import EpochSaver, TokenizedSkipgramDataGenerator, clean_text, fix_authors, tokenize_sets
```