extends RefCounted
class_name PipeIPC

# Thin helper around OS.execute_with_pipe that reads stdout/stderr on a background
# thread and exposes a pollable queue for the main thread.

signal stderr_line(line: String)
signal exited

var _process_io: FileAccess
var _stderr_io: FileAccess
var _pid: int = -1
var _thread: Thread
var _running := false

var _queue: Array[String] = []
var _mutex := Mutex.new()


func open(command: String, args: PackedStringArray = PackedStringArray()) -> bool:
	close()
	var result = OS.execute_with_pipe(command, args, false)
	if typeof(result) != TYPE_DICTIONARY or not result.has("stdio"):
		return false

	_process_io = result["stdio"]
	_stderr_io = result.get("stderr", null)
	_pid = int(result.get("pid", -1))
	_running = true

	_thread = Thread.new()
	_thread.start(Callable(self, "_reader_loop"))
	return true


func send_lines(rows: PackedStringArray, meta_line: String = "") -> void:
	if not _running or _process_io == null:
		return
	if meta_line != "":
		_process_io.store_string(meta_line + "\n")
	for row in rows:
		_process_io.store_string(row.strip_edges(true, true) + "\n")
	_process_io.store_string("\n")
	_process_io.flush()


func poll() -> Array[String]:
	_mutex.lock()
	var items := _queue.duplicate()
	_queue.clear()
	_mutex.unlock()
	return items


func close() -> void:
	_running = false
	if _thread:
		_thread.wait_to_finish()
		_thread = null
	_process_io = null
	_stderr_io = null
	_pid = -1
	_queue.clear()


func is_running() -> bool:
	return _running and _pid != -1 and OS.is_process_running(_pid)


func _reader_loop() -> void:
	while _running:
		var had_data := false

		if _process_io and _process_io.get_available_bytes() > 0:
			var line := _process_io.get_line().strip_edges()
			if line != "":
				_mutex.lock()
				_queue.append(line)
				_mutex.unlock()
			had_data = true

		if _stderr_io and _stderr_io.get_available_bytes() > 0:
			var err_line := _stderr_io.get_line().strip_edges()
			if err_line != "":
				call_deferred("_emit_stderr", err_line)
			had_data = true

		if not had_data:
			if _pid != -1 and not OS.is_process_running(_pid):
				break
			OS.delay_msec(5)

	_running = false
	call_deferred("_emit_exit")


func _emit_stderr(line: String) -> void:
	stderr_line.emit(line)


func _emit_exit() -> void:
	exited.emit()
