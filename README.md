# Notebook PGE

### Generating a base Notebook PGE project
```bash
$ notebook-pge-wrapper create <project_name>
```
The following project structure will be generated
```
.
├── Dockerfile.template
├── README.md
├── docker
│   ├── Dockerfile
│   └── requirements.ipynb
├── notebook_pges
│   └── test_nb_sample_pge.ipynb
├── pele_setup.ipynb
├── pge_create.ipynb
└── submit_job.ipynb
```
* `Dockerfile` will go through [container-builder](https://github.com/hysds/container-builder) to build the docker image
    * Will later be used to execute the notebook in a PGE setting
* Place all your `.ipynb` files in `notebook_pges/`


### Dockerfile
The `notebook-pge-wrapper dockerfile` command will read in values from `settings.yml` and fill in the values in 
`Dockerfile.template`

To use a new docker image edit the `base_image` value in `settings.yml`

```yaml
base_image: artifactory.com/nisar_ade:r1.3
user: jovyan
```

To update the `Dockerfile` with a new `base_image` run this command in the ***root*** directory of your project
```bash
$ notebook-pge-wrapper dockerfile
```


### Papermill
Your notebooks will leverage `papermill` to execute notebooks
* [Papermill Project](https://papermill.readthedocs.io/en/latest/)
* [Papermill parameters documentation](https://papermill.readthedocs.io/en/latest/usage-parameterize.html)
* Can set parameter types 2 ways
    * Adding a comment, ie. `x = 34  # type: int`
    * [Python type hinting](https://docs.python.org/3/library/typing.html) (introduced in python 3.5)

### Requirements.ipynb
`requirements.ipynb` is used to install additional dependencies in the docker image

It's separated into 3 sections:
* CentOS installation (`sudo yum install <package> -y`)
* Conda installtion (`sudo conda install <package> -y`)
* Python library installation (`sudo pip install <package>`)

### HySDS job specs generation
* Tag the top notebook cell with `parameters`
    * To add tags in your notebook, go to the toolbar: `View` -> `Cell Toolbar` -> `Tags`
* prepend any hysds specification fields with `_` and `extract_hysds_specs` will populate `hysds-io` and
`job_specs` with it's specified values
    * If not found it will set to `default` values (ie. `time_limit: 3600`)
```
# job_specs
_time_limit -> time_limit
_soft_time_limit -> soft_time_limit
_disk_usage -> disk_usage
_required_queue -> required_queue

# hysds-ios
_submission_type -> submission_type
_label -> label
```   
Example:
```python
# parameters
from typing import List

a = 100 # type: int
b = "jfksl" # type: str
c = 10.2553 # type: float
d = [1,2,3,4,5] # type: List
e: List[str] = ['a', 'b', 'c']
f = "fjskl"
g: List[str] = ["a", "b", "c"]

# hysds specs
_time_limit = 57389
_soft_time_limit = 4738
_disk_usage = "10GB"
_submission_type = "iteration"
_required_queue = "test_queue-worker"
_label = "TEST LABEL FOR HYSDS_IOS"
``` 

Use the `notebook-pge-wrapper` cli to generate the spec files
* `notebook-pge-wrapper specs all` to iterate generate spec files for all notebooks in `notebook_pges/`
* `notebook-pge-wrapper specs <notebook path>` to generate spec files for a notebook
```bash
$ notebook-pge-wrapper specs --help
Usage: notebook-pge-wrapper specs [OPTIONS] NOTEBOOK_PATH

  Generates the hysdsio and job specs for json files (in the docker
  directory) for a notebook

  enter "all" to generate all spec files in notebook_pges/

  ie. notebook-pge-wrapper specs <notebook_path or all>

Options:
  -s, --settings TEXT  (optional) path to settings.yml, will default to
                       ~/.config/notebook-pge-wrapper/settings.yml if not
                       supplied

  --help               Show this message and exit.
```
