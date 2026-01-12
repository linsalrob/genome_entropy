"""Basic tests to verify package installation and structure."""

import orf_entropy


def test_version():
    """Test that version is defined."""
    assert hasattr(orf_entropy, "__version__")
    assert isinstance(orf_entropy.__version__, str)


def test_package_import():
    """Test that the package can be imported."""
    assert orf_entropy is not None
