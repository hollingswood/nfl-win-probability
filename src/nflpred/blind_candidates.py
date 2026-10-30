"""Live hooks for registered blind candidates (blind.py): name -> fn(predictions) -> [{game_id, kickoff, home_win_prob}].
Only registered names may appear here (test_blind enforces it); each logs before kickoff on every full run."""
LIVE: dict = {}
