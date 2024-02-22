import dotenv
import pytest
import os


@pytest.fixture(scope="session", autouse=True)
def get_env():
    dotenv.load_dotenv(dotenv_path=".env.dev")


# This hook is used to add the images to the pytest-html report
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    dotenv.load_dotenv(dotenv_path=".env.dev")
    pytest_html = item.config.pluginmanager.getplugin("html")
    outcome = yield
    report = outcome.get_result()
    extras = getattr(report, "extras", [])
    if report.when == "call" or report.when == "setup":
        xfail = hasattr(report, "wasxfail")
        if not report.skipped and not xfail:
            # get folder of imgs_dir
            folder = os.path.split(os.getenv("UNIT_TESTING_IMGS_DIR"))[-1]
            file_name = report.nodeid.replace("/", "_") + ".png"
            file_path = os.path.join(folder, file_name)
            extras.append(pytest_html.extras.image(file_path, "logged image"))
        report.extras = extras
