extends Node

@warning_ignore("unused_signal")
signal chaser_won
@warning_ignore("unused_signal")
signal runner_won

var tiles := {
	"EMPTY": Vector2i(-1,-1),
	"BLOCK": Vector2i(0,0),
	"CHASER": Vector2i(0,1),
	"RUNNER": Vector2i(1,1),
	"EXIT": Vector2i(2,0),
}
