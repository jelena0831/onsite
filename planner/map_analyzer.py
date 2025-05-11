import json

def analyze_semantic_map(map_file_path):
    """分析语义地图文件，提取polygon对应的路径配置"""
    with open(map_file_path, 'r') as f:
        semantic_map = json.load(f)
    
    # 存储每个polygon的路径配置
    polygon_paths = {}
    
    # 遍历所有polygon
    for polygon in semantic_map['polygon']:
        polygon_token = polygon['token']
        paths = polygon['link_referencepath_tokens']
        
        # 按连接关系组织路径
        connected_paths = []
        current_sequence = []
        
        for path_token in paths:
            path_info = next(p for p in semantic_map['reference_path'] 
                           if p['token'] == path_token)
            if not current_sequence:
                current_sequence.append(path_token)
            else:
                # 检查是否与当前序列相连
                last_path = current_sequence[-1]
                last_path_info = next(p for p in semantic_map['reference_path'] 
                                    if p['token'] == last_path)
                if path_token in last_path_info['outgoing_tokens']:
                    current_sequence.append(path_token)
                else:
                    connected_paths.append(current_sequence)
                    current_sequence = [path_token]
        
        if current_sequence:
            connected_paths.append(current_sequence)
            
        if connected_paths:
            polygon_paths[polygon_token] = connected_paths
            
    return polygon_paths