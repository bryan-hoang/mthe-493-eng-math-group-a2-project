# mthe-493-group-a2

> Optimization of Data Allocation with Training Time Constraints in Heterogeneous Edge Learning

Main repo for MTHE 493 Group A2's documentation and source code. An implementation of a distributed edge learning algorithm that optimizes how data is allocated to workers.

## Getting started

### Windows/Mac/Linux

1. Ensure python 3.9 is installed.
2. (optional) Set up a python virtual environment (e.g. using `venv` or `conda`) and activate it.
3. Ensure `pip` is up to date: `pip install --upgrade pip`
4. Install dependencies: `pip install -r requirements.txt`

#### Usage

1. On any machine on the network, start notice board: `python src/notice_board.py` (in our testing, this was usually the orchestrator).
2. On each learner/worker, start worker script: `python src/worker.py`. (This usually doesn't work, since Axon doesn't handle discovery well. Fix: manually specify the notice board machine's LAN IP by running `python src/worker.py --nb-ip NB_IP`, replacing `NB_IP`)
3. You should see workers signing in on the notice board output.
4. On client, set the environment variables as needed (configures system parameters, cost, ML-related parameters, logging parameters) via `export KEY=VALUE`; or by configuring a `.env` file with the key-value pairs, one `KEY=VALUE` per line. See the section "Environment Variables" for details.
5. On client, run the client script: `python src/client.py`. (This usually doesn't work, since Axon doesn't handle discovery well. Fix: manually specify the notice board machine's LAN IP by running `python src/client.py --nb-ip NB_IP`, replacing `NB_IP`)

## Developer Setup

Same install instructions as above, except run `pip install -r requirements-dev.txt`.

**Notes about Gurobi:**

- If you want to run `src/data_assignment/assign_gurobi.py`, or compare allocation methods using `src/data_assignment/assign_test.py`, you must install Gurobi on your system (e.g. under a free academic license)
- Instructions can be found on Gurobi's website
- We specifically installed `gurobi` via a Conda environment through the Conda package manager. See [here](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-) for details.

**Notes about optimizers and `data_assignment` testing**

- The module `src/data_assignment/assign_test.py` will simulate performance of each of the 3 optimizer implementations (heuristic, PuLP, and Gurobi).
- This is a module, not a script: you must import it into a Python shell session or script and use the functions there.
- **Most importantly: the actual client script is only written to support the heuristic algorithm, because it performs the best based on tool selection/performance evaluation**. This can be changed easily (by adding a new environment variable in `environment.py`, and modifying the `client.py` script where it deals with data assignment)

Example usage:

```python
from src.data_assignment.assign_test import *

# Arguments
# num of tests to run
n_tests = 100000
# random seed, for reproducibility
seed = 1
# how much info should be printed/logged? options: 0, 1, 2
verbose = 0
# should we dump logs to file?
log = True
# should we log results of every test case to file?
log_all = False

# This is the primary function call
# Options for optimizers can be changed. Use: run_tests_heuristic, run_tests_gurobi, run_tests_pulp
results, discrepancies = validate(n_tests, run_tests_heuristic, run_tests_pulp, seed=seed, verbose=verbose, log=log, log_all=log_all)

# You can now investigate results + discrepancies as desired.
```

**Notes about large test runs with parameter variations**

- The script `src/run_tests.py` can be modified to specify specific system parameter variations to try running in sequence.
- This is how we generated all the data for the thesis: determine which parameters need to be varied, modify the script, then run it and wait
- The script generates all possible combinations of parameter values specified (cartesian product), then runs the client script with each variation, recording the results.
- This script sometimes also requires the specification of noticeboard IP, i.e. `python src/run_tests.py --nb-ip NB_IP`
- It will run each test, dump a single log to the `logs/` directory for each test, then (upon all tests completing) will dump a `dump.json` file containing all data. The JSON file is very easy to work with for data visualization; see the jupyter notebooks in the project for examples

**Notes about setting up workers for easy access**

- I set up SSH access to all computers for the purpose of logging in/out of them with shell access quickly
- This was the easiest way to run many tests centrally, and record all relevant data on one machine:
  - My computer acts as notice board + orchestrator (client)
  - SSH into all machines from my computer, run `worker` script
  - Run `client.py` or `run_tests.py`, analyze the output data
- Easy to repeat and reproduce this setup

## Environment Variables

| Name                  | Value(s)                                   | Default             | Description                                                                 |
| --------------------- | ------------------------------------------ | ------------------- | --------------------------------------------------------------------------- |
| BETA                  | int                                        | `1`                 | beta system parameter                                                       |
| S_MIN                 | int                                        | `10`                | s_min system parameter                                                      |
| MAX_TIME\*\*          | float                                      | `30.0`              | max runtime system parameter                                                |
| FEE_TYPE\*            | `random`, `constant`, `linear`, `specific` | `"constant"`        | Type of worker fee setup                                                    |
| FEES                  | comma-separated string of floats           | `"1.0,1.0,...,1.0"` | Fee values under `specific` FEE_TYPE. Padded to number of workers in system |
| DEFAULT_FEE           | float                                      | `1.0`               | Default fee, for `constant`, and for padding `specific` FEE_TYPE            |
| NUM_BENCHMARK         | int                                        | `1000`              | Number of fake batches to compute during worker benchmark                   |
| NUM_GLOBAL_CYCLES\*\* | int                                        | `10`                | Number of learning + aggregation cycles that are performed                  |
| BATCH_SIZE            | int                                        | `32`                | Number of samples in each batch                                             |
| WEIGHT_TYPE           | `xavier`, `kaiming`, `orthogonal`          | `"xavier"`          | How should weights be initialized for each worker?                          |
| ALLOW_GPU_DEVICE      | bool                                       | `True`              | Should workers use their GPU, if it is available?                           |
| LOGS                  | str                                        | `"logs"`            | Path to directory where logs are dumped                                     |

\*FEE_TYPE:

- `random` = random fees for each worker, between 1-20 (inclusive)
- `constant` = constant fees for all workers, equal to DEFAULT_FEE
- `linear` = fees are set to `1, 2, ..., n` for n workers
- `specific` = Specific fee structure, determined by FEES.
  - If FEES is of length less than n, remainder is padded to n, using DEFAULT_FEE
  - If FEES is of length greater than n, remainder is truncated to n

\*\*:

- There's a weird implementation detail here. `MAX_TIME` specifies the _total duration of time_ the system is allowed to take, and `NUM_GLOBAL_CYCLES` specifies how many learning cycles must occur in this time. Hence, each global update cycle must complete in `MAX_TIME` / `NUM_GLOBAL_CYCLES` seconds.
- Jack has indicated this is not the best design choice. This can be changed if needed.

**IMPORTANT**: Everything below this point is untested and/or out-of-date documentation.

## Getting started

For an overview of the project's architecture, refer to the [ARCHITECTURE.md](ARCHITECTURE.md).

### Raspberry Pi (not functional)

Note: The project dependencies target an environment that is a 64-bit ARM architecture (e.g., a raspberrypi) with python 3.9.

To install necessary project dependencies, run

```shell
git clone https://github.com/bryan-hoang/mthe-493-group-a2.git
cd mthe-493-group-a2
make install
```

on each machine to be used to install the python dependencies using `pipenv` and to ensure proper libraries are installed for `pytorch` to work.

#### Usage (Pi / Pipenv)

**Access the `pipenv` virtual environment using `pipenv shell` or `pipenv run <command>`**. Then to set up the system,

1. Start the notice board by running

   ```sh
   python src/notice_board.py
   ```

   on a machine on the network.

1. Start the workers by running

   ```sh
   python src/worker.py
   ```

   on each machine that you would like to use as a worker.

1. Configure the client's parameters by running

   ```sh
   dotenv set BETA <value>
   dotenv set S_MIN <value>
   ```

   on the machine you would like to act as the orchestrator. See [Configuration](#configuration) for more details.

1. Start the client by running

   ```sh
   python src/client.py
   ```

   on the machine you would like to act as the orchestrator.

## Developing

### Built With

- [axon-ecrg](https://github.com/DuncanMays/axon-ECRG)
- [pytorch](https://github.com/pytorch/pytorch)

### Prerequisites

- `python 3.9`

### Setting up Dev

```shell
git clone https://github.com/bryan-hoang/mthe-493-group-a2.git
cd mthe-493-group-a2
make install-dev
```

The `install-dev` recipe installs additional useful development experience packages list in the project's [Pipfile](Pipfile).

### Deploying / Publishing

<!-- TODO(bryan-hoang): Not sure if/how we're planning on improving this part of the orkflow. -->

give instructions on how to build and release a new version
In case there's some step you have to take that publishes this project to a
server, this is the right time to state it.

```shell
packagemanager deploy your-project -s server.com -u username -p password
```

And again you'd need to tell what the previous code actually does.

## Configuration

The project uses [python-dotenv](https://github.com/theskumar/python-dotenv#python-dotenv) to load environment variables from a `.env` file the [src/client.py](src/client.py) reads from to retrieve the `BETA` and `S_MIN` parameters.

The package has a [CLI command-line interface](https://github.com/theskumar/python-dotenv#command-line-interface) to make settings the values in the `.env` file easier.

## Tests

The project is set up to use [pytest](https://github.com/pytest-dev/pytest#readme) to detect and run all test files. `make test` is a recipe that will run the tests naively under the [src/tests](src/tests) folder.

```shell
make test
```

## Style guide

[Black](https://github.com/psf/black).
