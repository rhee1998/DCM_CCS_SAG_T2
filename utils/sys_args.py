#!/usr/bin/env python3
import sys

# ====================== #
# Parse System Arguments #
# ====================== #
def SysArgumentsToDict(argv):
    args_dict = {}
    
    for user_input in argv:
        if not '=' in user_input: continue
        
        var_name  = user_input.split('=')[0]
        var_value = user_input[(len(var_name) + 1):]
        args_dict[var_name] = var_value

    return args_dict