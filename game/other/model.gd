extends Node

const PipeIPC = preload("res://other/ipc.gd")

# Autoload that owns the external model process (Python RL agent) by delegating
# the pipe/thread work to PipeIPC.

signal action_received(action: String) # Fired when the model prints a line to stdout.
signal stderr_line(line: String)       # Fired for stderr lines (useful for debugging).
signal process_exited()                # Fired when the child process stops.

var _ipc: PipeIPC = PipeIPC.new()

@export var python_path: String = "python"
@export_file("*.py") var script_path: String = "res://../main.py"
@export_enum("catcher", "runner") var role: String = "catcher"
@export var catcher_index: int = 0
@export var allow_unknown: bool = false
@export var extra_args: PackedStringArray = []


func _ready() -> void:
	_ipc.stderr_line.connect(func(line: String) -> void: stderr_line.emit(line))
	_ipc.exited.connect(func() -> void: process_exited.emit())


# Start the external model. Command-line arguments are built here.
func start() -> void:
	var py := python_path
	var script_abs := ProjectSettings.globalize_path(script_path)
	var args: PackedStringArray = [
		script_abs,
		"pipe",
		"--role", role,
		"--catcher-index", str(catcher_index)
	]
	if allow_unknown:
		args.append("--allow-unknown")
	args.append_array(extra_args)

	var ok = _ipc.open(py, args)
	if not ok:
		push_warning("Failed to start model process: execute_with_pipe returned an unexpected result.")


# Send one observation to the model.
# `rows` should be the strings you already render for the vision map (see pipe protocol).
# An optional `meta_line` (e.g. "role=runner id=0") is sent first when provided.
func send_data(prefix: String = "", rows: PackedStringArray = PackedStringArray()) -> void:
	_ipc.send_lines(rows, prefix)


func stop() -> void:
	_ipc.close()


func is_running() -> bool:
	return _ipc.is_running()


func _process(_delta: float) -> void:
	for action in _ipc.poll():
		action_received.emit(action)
