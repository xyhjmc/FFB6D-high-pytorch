"""
和 SC 版本几乎一样，只是多传一个 cls_id：
outputs = model(rgb, pts_xyz, pts_uv_norm, cls_ids)
loss, loss_dict = self.loss_fn(outputs, batch)


"""

