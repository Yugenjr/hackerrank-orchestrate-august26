import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader

@pytest.fixture
def mock_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    # Create mock users.csv
    users_csv = dataset_dir / "users.csv"
    users_csv.write_text("user_id,do_not_disturb_window\nu1,22:00-07:00\nu2,09:00-17:00\n")
    
    # Create mock groups.csv
    groups_csv = dataset_dir / "groups.csv"
    groups_csv.write_text("group_id,group_name\ng1,Family\n")
    
    # Create mock group_members.csv
    group_members_csv = dataset_dir / "group_members.csv"
    group_members_csv.write_text("group_id,user_id,group_muted_by_user\ng1,u1,True\ng1,u2,False\n")
    
    return str(dataset_dir)

def test_data_loader_initialization(mock_dataset_dir):
    loader = DataLoader(dataset_dir=mock_dataset_dir)
    loader.load_all()
    
    # Test single index lookup
    user = loader.get_user("u1")
    assert user is not None
    assert user["do_not_disturb_window"] == "22:00-07:00"
    
    user_none = loader.get_user("u3")
    assert user_none is None
    
    # Test multi-index lookup
    member = loader.get_group_member("g1", "u1")
    assert member is not None
    assert member["group_muted_by_user"]
    
    # Test missing file graceful handling
    # (e.g. messages.csv wasn't created in the mock, should not crash if handled properly, though load_csv warns)
    msg = loader._get_row("messages.csv", "m1")
    assert msg is None
