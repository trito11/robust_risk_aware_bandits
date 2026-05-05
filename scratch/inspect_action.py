from simglucose.simulation.scenario import Action
import inspect

print(f"Action: {Action}")
try:
    a = Action(insulin=1, meal=10)
    print(f"Action created with keywords: {a}")
except Exception as e:
    print(f"Failed keywords: {e}")

try:
    a = Action(1, 10)
    print(f"Action created with positional: {a}")
except Exception as e:
    print(f"Failed positional: {e}")

if hasattr(Action, '_fields'):
    print(f"Fields: {Action._fields}")
