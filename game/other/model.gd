extends Node

#const PipeIPC = preload("res://other/ipc.gd")

# Autoload that owns the external model process (Python RL agent) by delegating
# the pipe/thread work to PipeIPC.

#Input format - Vector2i, String 
signal action_received(action: String) # Fired when the model prints a line to stdout.
signal stderr_line(line: String)       # Fired for stderr lines (useful for debugging).
signal process_exited()                # Fired when the child process stops.

var _ipc: PipeIPC = PipeIPC.new()

@export var conda_env_name: String = "lockeduprl"
@export var conda_executable: String = "/opt/miniconda3/condabin/conda"
@export_global_file("*.py") var script_path: String = "../main.py"
@export var extra_args: PackedStringArray = []


func _ready() -> void:
	_ipc.stderr_line.connect(func(line: String) -> void: stderr_line.emit(line))
	_ipc.exited.connect(func() -> void: process_exited.emit())


# Start the external model. Command-line arguments are built here.
func start() -> void:
	var project_root := ProjectSettings.globalize_path("res://")
	var script_abs := _to_abs_path(script_path, project_root)

	var conda := conda_executable
	if conda.begins_with("res://"):
		conda = ProjectSettings.globalize_path(conda)
	var cmdline := "cd %s && exec %s run -n %s --no-capture-output python %s pipe" % [
		_shell_quote(project_root),
		_shell_quote(conda),
		_shell_quote(conda_env_name),
		_shell_quote(script_abs)
	]
	if extra_args.size() > 0:
		for a in extra_args:
			cmdline += " " + _shell_quote(a)

	var ok = _ipc.open("/bin/bash", ["-c", cmdline])
	if not ok:
		push_warning("Failed to start model process: execute_with_pipe returned an unexpected result.")


func _to_abs_path(path: String, project_root: String) -> String:
	if path.begins_with("res://"):
		return ProjectSettings.globalize_path(path)
	if path.begins_with("/"):
		return path
	return project_root.path_join(path)


func _shell_quote(text: String) -> String:
	# Minimal shell quoting to preserve spaces; not handling every edge case.
	var escaped := text.replace("'", "'\"'\"'")
	return "'" + escaped + "'"


# Send one observation to the model.
# `rows` should be the strings you already render for the vision map (see pipe protocol).
# An optional `meta_line` (e.g. "role C1 pos 3 12") is sent first when provided.
func send_data(prefix: String = "", rows: PackedStringArray = PackedStringArray()) -> void:
	_ipc.send_lines(rows, prefix)


func stop() -> void:
	_ipc.close()


func is_running() -> bool:
	return _ipc.is_running()


func _process(_delta: float) -> void:
	for action : String in _ipc.poll():
		action_received.emit(action)
