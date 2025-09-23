def train_step(model, batch, optimizer, w_rec=1.0, w_roles=1.0):
    model.train()
    out = model(batch["tok_ids"], batch["pos_ids"], batch["feat_rows"], batch["lengths"],
                batch.get("edge_index"), batch.get("edge_type"))
    loss = 0.0
    if w_rec:
        loss += w_rec * model.loss_reconstruction(out["feat_hat"], batch["feat_rows"], out["mask"])
    if w_roles and "role_labels" in batch:
        loss += w_roles * model.loss_roles(out["role_logits"], batch["role_labels"], out["mask"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}