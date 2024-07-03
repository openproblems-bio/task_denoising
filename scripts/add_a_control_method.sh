#!/bin/bash

task_name="task_template"
component_name="my_control_method"
component_lang="python" # change this to "r" if need be

common/create_component/create_component \
  --task $task_name \
  --language "$component_lang" \
  --name "$component_name" \
  --api_file src/api/comp_control_method.yaml \
  --output "src/control_methods/$component_name"