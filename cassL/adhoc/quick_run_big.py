from cassL import user_interface as ui
import os

already_done = ["template", "Hnu2MEGA"]

for file in os.listdir("scenarios"):
    # The name, as user_interface expects, lacks the file ending
    scenario_name = file[:-4]
    
    if scenario_name in already_done:
        continue
    
    print(scenario_name)
    
    try:
        ui.build_train_and_test_sets(scenario_name)
    except Exception:
        print("There is something wrong with this scenario file!")
        
print("Done!")
