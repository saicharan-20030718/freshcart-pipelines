# Import Mage decorators
# If Mage is not installed, use simple fallback decorators.

try:
    from mage_ai.data_preparation.decorators import transformer, test
except ImportError:

    def transformer(func):
        return func

    def test(func):
        return func


from sklearn.model_selection import train_test_split


@transformer
def transform(df, *args, **kwargs):

    # Split the dataset into training and testing records
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=df["target"]
    )

    # Create a copy of the original DataFrame
    output = df.copy()

    # Initially mark all records as training data
    output["split"] = "train"

    # Mark the test records
    output.loc[test_indices, "split"] = "test"

    return output


@test
def test_output(output, *args):

    assert "split" in output.columns

    assert (output["split"] == "train").sum() == 120

    assert (output["split"] == "test").sum() == 30