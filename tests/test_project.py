def test_app_package_is_importable():
    import app

    assert app.__doc__ == "Cuy Monitor AI service application."
