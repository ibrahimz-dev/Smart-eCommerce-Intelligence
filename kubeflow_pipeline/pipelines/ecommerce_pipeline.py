from kfp import dsl


@dsl.pipeline(
    name="Smart E-Commerce Intelligence Pipeline",
    description="Scrape → Preprocess → Train → Score → Top-K",
)
def ecommerce_pipeline(
    store_urls: list[str] = None,
    top_k: int = 50,
):
    if store_urls is None:
        store_urls = ["https://www.kyliecosmetics.com", "https://shop.lego.com"]

    from kubeflow_pipeline.components.scraping_component import scrape_products, validate_scraped_data
    from kubeflow_pipeline.components.preprocessing_component import preprocess_data
    from kubeflow_pipeline.components.training_component import train_classification_models, train_clustering_models
    from kubeflow_pipeline.components.scoring_component import calculate_product_score
    from kubeflow_pipeline.components.topk_component import select_top_k_products, generate_bi_report

    scrape_task = scrape_products(store_urls=store_urls)
    validate_task = validate_scraped_data(data_path=scrape_task.outputs["output_path"])
    preprocess_task = preprocess_data(data_path=scrape_task.outputs["output_path"])
    train_cls = train_classification_models(
        train_data=preprocess_task.outputs["train_data"],
        val_data=preprocess_task.outputs["val_data"],
        test_data=preprocess_task.outputs["test_data"],
    )
    train_clu = train_clustering_models(train_data=preprocess_task.outputs["train_data"])
    score_task = calculate_product_score(data_path=scrape_task.outputs["output_path"])
    top_k_task = select_top_k_products(
        scored_data=score_task.outputs["scored_path"],
        raw_data=scrape_task.outputs["output_path"],
        k=top_k,
    )
    generate_bi_report(
        top_k_data=top_k_task.outputs["top_k_list"],
        clustering_data=train_clu.outputs["cluster_profiles_path"],
    )


if __name__ == "__main__":
    import kfp
    kfp.compiler.Compiler().compile(
        pipeline_func=ecommerce_pipeline,
        package_path="ecommerce_pipeline.yaml",
    )
    print("Pipeline compiled to ecommerce_pipeline.yaml")
