import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


class AssociationRuleMiner:
    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.3,
                 min_lift: float = 1.0, max_len: int = 4):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_len = max_len
        self.frequent_itemsets = None
        self.rules = None

    def prepare_transactions(self, df: pd.DataFrame, group_col: str = "shop_name",
                              item_col: str = "category") -> list:
        transactions = df.groupby(group_col)[item_col].apply(list).tolist()
        return transactions

    def create_basket_matrix(self, transactions: list) -> pd.DataFrame:
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        return pd.DataFrame(te_ary, columns=te.columns_)

    def fit(self, df: pd.DataFrame, group_col: str = "shop_name",
            item_col: str = "subcategory"):
        transactions = self.prepare_transactions(df, group_col, item_col)
        basket = self.create_basket_matrix(transactions)

        self.frequent_itemsets = apriori(
            basket,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
        )

        if not self.frequent_itemsets.empty:
            self.rules = association_rules(
                self.frequent_itemsets,
                metric="lift",
                min_threshold=self.min_lift,
            )

            self.rules = self.rules[
                self.rules["confidence"] >= self.min_confidence
            ].sort_values("lift", ascending=False).reset_index(drop=True)

        return self

    def get_top_rules(self, n: int = 20) -> pd.DataFrame:
        if self.rules is None or self.rules.empty:
            return pd.DataFrame()
        return self.rules.head(n)

    def get_rules_for_antecedent(self, item: str, n: int = 10) -> pd.DataFrame:
        if self.rules is None:
            return pd.DataFrame()
        mask = self.rules["antecedents"].apply(
            lambda x: item.lower() in str(x).lower()
        )
        return self.rules[mask].head(n)

    def get_summary(self) -> dict:
        if self.rules is None:
            return {"n_rules": 0}
        return {
            "n_rules": len(self.rules),
            "n_frequent_itemsets": len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
            "avg_support": round(self.rules["support"].mean(), 4),
            "avg_confidence": round(self.rules["confidence"].mean(), 4),
            "avg_lift": round(self.rules["lift"].mean(), 4),
            "min_support_param": self.min_support,
            "min_confidence_param": self.min_confidence,
            "min_lift_param": self.min_lift,
        }

    def rules_to_readable(self, n: int = 10) -> pd.DataFrame:
        if self.rules is None or self.rules.empty:
            return pd.DataFrame()

        top_rules = self.rules.head(n).copy()
        readable = pd.DataFrame({
            "rule": top_rules.apply(
                lambda r: f"{{{', '.join(list(r['antecedents']))}}} -> {{{', '.join(list(r['consequents']))}}}",
                axis=1
            ),
            "support": top_rules["support"].round(4),
            "confidence": top_rules["confidence"].round(4),
            "lift": top_rules["lift"].round(4),
        })
        return readable
