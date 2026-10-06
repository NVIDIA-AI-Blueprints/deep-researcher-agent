# OpenShell Policies

`deep-researcher-research-policy.yaml` is the checked-in, production-oriented policy sample
used by schema and regression tests. It is not the runtime default.

Run `scripts/openshell/setup_openshell.sh` to generate the canonical runtime policy at:

```text
configs/openshell/generated/deep-researcher-openshell-policy.yaml
```

The generated file reflects the selected network preset and Landlock mode and is
the default consumed by `configs/config_openshell.yml`, the strict gateway probe,
and the live acceptance suite. Do not commit environment-specific generated
policies or credentials.
