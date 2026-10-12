import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go


class PCAAnalyzer:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.X_pca = None
        self.explained_variance_ = None
        self.loadings_ = None

    def fit(self, X: np.ndarray, feature_names: list = None):
        X_scaled = self.scaler.fit_transform(X)
        self.X_pca = self.pca.fit_transform(X_scaled)
        self.explained_variance_ = self.pca.explained_variance_ratio_

        if feature_names is not None:
            self.loadings_ = pd.DataFrame(
                self.pca.components_.T,
                columns=[f"PC{i+1}" for i in range(self.n_components)],
                index=feature_names[:X.shape[1]],
            )
        return self

    def get_2d_projection(self, df_original: pd.DataFrame = None,
                          color_col: str = "category") -> go.Figure:
        if self.X_pca is None or self.X_pca.shape[1] < 2:
            return go.Figure()

        plot_df = pd.DataFrame({
            "PC1": self.X_pca[:, 0],
            "PC2": self.X_pca[:, 1],
        })

        if df_original is not None:
            for col in df_original.columns:
                if len(df_original[col].values) == len(plot_df):
                    plot_df[col] = df_original[col].values

        var_pct1 = round(self.explained_variance_[0] * 100, 1)
        var_pct2 = round(self.explained_variance_[1] * 100, 1)

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color=color_col if color_col in plot_df.columns else None,
            hover_data=["name", "price", "rating"] if "name" in plot_df.columns else None,
            title=f"PCA Product Space (PC1: {var_pct1}%, PC2: {var_pct2}%)",
            labels={"PC1": f"Principal Component 1 ({var_pct1}%)",
                    "PC2": f"Principal Component 2 ({var_pct2}%)"},
            opacity=0.7,
        )

        fig.update_layout(
            template="plotly_white",
            width=900,
            height=600,
            legend_title_text="Category",
        )
        return fig


    def get_loadings_chart(self, top_n: int = 15) -> go.Figure:
        if self.loadings_ is None:
            return go.Figure()

        loadings_abs = self.loadings_.abs().sum(axis=1).sort_values(ascending=False)
        top_features = loadings_abs.head(top_n).index

        fig = go.Figure()
        for pc in self.loadings_.columns[:2]:
            fig.add_trace(go.Bar(
                name=pc,
                x=top_features,
                y=self.loadings_.loc[top_features, pc],
            ))

        fig.update_layout(
            title="Top Feature Contributions to Principal Components",
            xaxis_title="Features",
            yaxis_title="Loading Value",
            barmode="group",
            template="plotly_white",
            height=500,
        )
        return fig

    def get_summary(self) -> dict:
        if self.explained_variance_ is None:
            return {}
        return {
            "explained_variance_ratio": [round(v, 4) for v in self.explained_variance_],
            "cumulative_variance": [round(sum(self.explained_variance_[:i+1]), 4)
                                     for i in range(len(self.explained_variance_))],
            "n_components": self.n_components,
        }
