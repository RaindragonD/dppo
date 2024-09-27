
import os
import html
import numpy as np

def generate_html(idx):
    base_dir = "/home/yl/dppo/log/robomimic-eval/square_eval_diffusion_mlp_ta8_td20"
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluation Videos for Checkpoint {}</title>
        <style>
            .video-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                padding: 20px;
            }}
            .video-container {{
                border: 5px solid;
                border-radius: 10px;
                overflow: hidden;
            }}
            .success {{ border-color: #4CAF50; }}
            .failure {{ border-color: #F44336; }}
            video {{
                width: 100%;
                display: block;
            }}
            .video-info {{
                padding: 10px;
                text-align: center;
                font-family: Arial, sans-serif;
            }}
        </style>
    </head>
    <body>
        <h1>Evaluation Videos for Checkpoint {}</h1>
        <div class="video-grid">
    """.format(idx, idx)

    for iteration in range(100):
        results = np.load(os.path.join(base_dir, str(iteration), "result.npz"))
        rewards = results["reward_trajs_split"]
        render_dir = os.path.join(base_dir, str(iteration), "render")
        if not os.path.exists(render_dir):
            continue
        success = np.max(rewards[idx]) >= 1
        status_class = "success" if success else "failure"
        
        html_content += f"""
            <div class="video-container {status_class}">
                <video controls>
                    <source src="{iteration}/render/{html.escape(f"trial-{idx}_reset-0.mp4")}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="video-info">
                    Iteration: {iteration}, Env ID: {idx}<br>
                    Status: {"Success" if success else "Failure"}
                </div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    html_file_path = os.path.join(base_dir, f"evaluation_videos_{idx}.html")
    with open(html_file_path, "w") as f:
        f.write(html_content)

    print(f"HTML file with video grid created at: {html_file_path}")

if __name__ == "__main__":
    for checkpoint_itr in range(100):
        os.system(f"python script/run.py --config-name=eval_diffusion_mlp_ft --config-dir=cfg/robomimic/eval/square checkpoint_itr={checkpoint_itr}")
    for idx in range(4):
        generate_html(idx)