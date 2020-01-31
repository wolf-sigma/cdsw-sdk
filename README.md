# Utilities for [Cloudera Data Science Workbench]https://www.cloudera.com/products/data-science-and-engineering/data-science-workbench.html)

This package allows the user to

 - launch and stop worker engines
 - track files and metrics in experiments and models
 - test models from sessions
 - get authentication parameters to be used with the REST API

from within Python.

## Installation

This package is preinstalled on CDSW. You can import it with

```python
import cdsw
```

## Testing

To test the `track_metric` function, run:

```
/app/services/web/api/altus-ds-1/controllers/batch-runs# mocha test.js --grep metrics
```

in a web pod. To test launchers, run:

```
/app/services/web# gulp test:workload --grep python[23]-workers
```

More automated testing is planned. See DSE-6235. In the meantime, for manual
testing of track_metric, you can run the following as an experiment.

```
import cdsw

def track(label, value):
  try:
    cdsw.track_metric(label, value)
  except ValueError as e:
    print(e)
    print("Did not log", label)

track("X", 1)           # Should track in both python versions
track(u"Unicode X", 1)  # Should track in both python versions
track("π", 3.14)        # Should track in both python versions
track(1, 1)             # Should track in either python version
track(b"Bytes X", 1)    # Should track in python2 only
```

## Rich engine output

IPython provides its own [rich display system](http://nbviewer.ipython.org/urls/raw.github.com/ipython/ipython/1.x/examples/notebooks/Part%205%20-%20Rich%20Display%20System.ipynb).
You can use IPython's rich display system to display HTML, images and more in a session or job.

Note that in html tags produced by IPython, relative paths are relative to the CDN; for example, `folder/image.png` resolves to the file in the engine's cdn folder at `/cdn/folder/image.png`.

#### Note: Multi-command matplotlib plots

It's common to generate plots using multiple commands, for example

```python
import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.title("Some numbers")
```

By default, each of these commands will generate a new plot. To make all commands apply to the current plot, use the magic command

```python
%config InlineBackend.close_figures = False
```

Each command will still re-render the plot. To create a multi-command plot and have it appear only once, simply wrap them in a function or `with` statement.

# License

Copyright (c) Cloudera Inc. 2017, All Rights Reserved
