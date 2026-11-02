"""The dumps' atomic replace: tried again while Windows refuses it (a reader has the frame open)."""
import os
import pytest
from gfdtd import dumps


def test_replace_waits_for_a_reader(tmp_path, monkeypatch):
    tmp, path = tmp_path / 'f.part', tmp_path / 'f.vtr'
    tmp.write_text('new', encoding='utf-8')
    path.write_text('old', encoding='utf-8')
    real, refused = os.replace, []

    def flaky(a, b):
        if len(refused) < 3:                                        # the GUI holds the file for three tries
            refused.append(1)
            raise PermissionError(13, 'in use')
        return real(a, b)
    monkeypatch.setattr(dumps.os, 'replace', flaky)
    dumps.replace(str(tmp), str(path))
    assert path.read_text(encoding='utf-8') == 'new' and len(refused) == 3 and not tmp.exists()


def test_replace_gives_up(tmp_path, monkeypatch):
    def refuse(a, b):
        raise PermissionError(13, 'in use')
    monkeypatch.setattr(dumps.os, 'replace', refuse)
    monkeypatch.setattr('time.sleep', lambda s: None)
    with pytest.raises(PermissionError):
        dumps.replace('a', 'b', tries=3)
