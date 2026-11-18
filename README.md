# D-AFC

D-AFC trains closed-loop jet-control policies for two-dimensional cylinder and square-cylinder flows using differentiable simulation. The Python package is `myafc`.

## Installation

The [environment files](0a-env/) record the Python and simulation dependencies. For example, on Linux with CUDA 12.1, run these commands from the repository root:

```bash
conda env create -f 0a-env/env_all_linux-zero_phi310.yaml
conda activate zero_phi310
python -m pip install -e ./0dev_ing/phiflow_based/0mydev/myafc
```

## Examples

The [examples directory](0dev_ing/phiflow_based/0mydev/myafc/examples/) contains the solver and trainer entry points and their YAML configurations. From the repository root, inspect the training options with:

```bash
cd 0dev_ing/phiflow_based/0mydev/myafc/examples
python run_jet_trainer/train_jet.py --help
```

From that directory, launch the circular-cylinder example with the matching trainer and flow configurations:

```bash
python run_jet_trainer/train_jet.py run_jet_trainer/cfgs/jet_trainer_main_sphere_iter.yaml \
    --env cfgs4env/karman_torch_drl_sphere.yaml
```

The square-cylinder example uses the corresponding `square` files. The package also includes [technical documentation](0dev_ing/phiflow_based/0mydev/myafc/src/myafc/docs/).
