import os
import gradio as gr
import embedding
from pathlib import Path


def inspect_tensors(embedding_name: str):
    model_path = os.path.join(embedding.EMBEDDINGS_DIR, embedding_name)
    if not os.path.isfile(model_path):
        return f"File not found: `{model_path}`."
    state_dict = embedding.load_model(model_path)
    model_type = embedding.get_model_type(state_dict)

    info = ""
    if model_type == 'XL':
        for key, weight in state_dict.items():
            info += f"{key}: \n{weight}\n"
    elif model_type == '1.5':
        weight = state_dict['string_to_param']['*']
        info += f"clip_l: \n{weight}"

    return info


def merge_embeddings(model_name_A, model_name_B, alpha, output_filename, save_as_safetensors, allow_overwrite_output):
    model_path_A = os.path.join(embedding.EMBEDDINGS_DIR, model_name_A)
    model_path_B = os.path.join(embedding.EMBEDDINGS_DIR, model_name_B)
    output_filepath = os.path.join(embedding.EMBEDDINGS_DIR, output_filename)

    embedding.merge_embeddings(
        model_path_A,
        model_path_B,
        alpha,
        output_filename=output_filepath,
        save_as_safetensors=save_as_safetensors,
        allow_overwrite_output=allow_overwrite_output
    )

    return f"Merged!\nSaved at `{output_filepath}`."


def merge_embedding_list(
    model_name_1,
    model_name_2,
    model_name_3,
    model_name_4,
    model_name_5,
    model_name_6,
    model_name_7,
    model_name_8,

    boost_1,
    boost_2,
    boost_3,
    boost_4,
    boost_5,
    boost_6,
    boost_7,
    boost_8,

    output_filename,
    save_as_XL,
    save_as_safetensors,
    allow_overwrite_output
):
    def name_to_path(filename):
        if len(filename) == 0:
            return None
        return os.path.join(embedding.EMBEDDINGS_DIR, filename)
    inp_model_paths = [
        name_to_path(model_name_1),
        name_to_path(model_name_2),
        name_to_path(model_name_3),
        name_to_path(model_name_4),
        name_to_path(model_name_5),
        name_to_path(model_name_6),
        name_to_path(model_name_7),
        name_to_path(model_name_8),
    ]
    inp_boosts = [
        boost_1,
        boost_2,
        boost_3,
        boost_4,
        boost_5,
        boost_6,
        boost_7,
        boost_8,
    ]
    model_paths = []
    boosts = []
    for i, model_path in enumerate(inp_model_paths):
        if model_path and os.path.isfile(model_path):
            model_paths.append(model_path)
            boosts.append(inp_boosts[i])

    if output_filename == '':
        param_str = '+'.join([f"{Path(model_paths[i]).stem}_x_{boosts[i]}" for i in range(len(model_paths))])
        output_filename = f"merged_({param_str.replace('.', '_')})"
    output_filepath = os.path.join(embedding.EMBEDDINGS_DIR, output_filename)

    embedding.merge_embedding_list(
        model_paths,
        boosts,
        as_type='XL' if save_as_XL else '1.5',
        output_filename=output_filepath,
        save_as_safetensors=save_as_safetensors,
        allow_overwrite_output=allow_overwrite_output
    )

    return f"Merged!\nSaved at `{output_filepath}`."


def convert_to_XL_embedding(model_name, output_filename, save_as_safetensors, allow_overwrite_output):
    model_path = os.path.join(embedding.EMBEDDINGS_DIR, model_name)
    output_filepath = os.path.join(embedding.EMBEDDINGS_DIR, output_filename)

    embedding.convert_embedding_as(
        model_path,
        output_filename=output_filepath,
        save_as_safetensors=save_as_safetensors,
        allow_overwrite_output=allow_overwrite_output
    )

    return f"Converted!\nSaved at `{output_filepath}`."


def create_ui():
    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row(variant='panel'):
                    save_as_XL = gr.Checkbox(
                        value=True,
                        label="Save as XL",
                        container=False,
                        scale=0
                    )
                    save_as_safetensors = gr.Checkbox(
                        value=True,
                        label="Save as safetensors",
                        container=False,
                        scale=0
                    )
                    allow_overwrite_output = gr.Checkbox(
                        value=False,
                        label="Allow overwrite",
                        container=False,
                        scale=0,
                    )
                    output_filename = gr.Textbox(
                        label="Output filename",
                        scale=1,
                        min_width=256
                    )
                with gr.Row():
                    merge_button = gr.Button(
                        value="Merge",
                        variant='primary'
                    )
                    # convert_button = gr.Button(
                    #     value="Convert to XL",
                    #     variant="primary"
                    # )

                with gr.Row():
                    state = gr.TextArea(
                        show_label=False,
                        interactive=False,
                        lines=3,
                    )

                with gr.Row():
                    refresh_button = gr.Button(
                        value="Refresh",
                        scale=0,
                        min_width=92,
                    )

                def create_input(idx: int = 0):
                    with gr.Row(equal_height=True, variant='compact'):
                        gr.Markdown(value=f"Embedding {idx}")
                        model_i = gr.Dropdown(
                            label=f"Embedding {idx}",
                            choices=os.listdir(embedding.EMBEDDINGS_DIR),
                            multiselect=False,
                            allow_custom_value=False,
                            container=False,
                            min_width=256,
                            scale=0
                        )
                        boost_i = gr.Number(
                            label="Boost",
                            value=1,
                            container=False,
                            scale=0,
                            min_width=96
                        )

                    return (model_i, boost_i)

                with gr.Row():
                    gr.Markdown("Embeddings")

                inputs = [create_input(i+1) for i in range(8)]

            with gr.Column(scale=1):
                with gr.Row(equal_height=True):
                    filename = gr.Textbox(
                        label="Embedding name",
                    )
                    inspect_button = gr.Button(
                        value="Inspect",
                        scale=0,
                        min_width=144,
                        variant='primary'
                    )

                with gr.Row():
                    tensors = gr.TextArea(
                        label="Tensors",
                        lines=20,
                        max_lines=20
                    )

        inspect_button.click(
            fn=inspect_tensors,
            inputs=[filename],
            outputs=[tensors]
        )

        refresh_button.click(
            fn=lambda: (gr.update(choices=os.listdir(embedding.EMBEDDINGS_DIR)), )*len(inputs),
            outputs=[*[inp[0] for inp in inputs]],
        )

        merge_button.click(
            fn=merge_embedding_list,
            inputs=[
                *[inp[0] for inp in inputs],  # Model paths
                *[inp[1] for inp in inputs],  # Boosts
                output_filename,
                save_as_XL,
                save_as_safetensors,
                allow_overwrite_output
            ],
            outputs=[state]
        )

        # convert_button.click(
        #     fn=convert_to_XL_embedding,
        #     inputs=[
        #         model_A,
        #         output_filename,
        #         save_as_safetensors,
        #         allow_overwrite_output
        #     ],
        #     outputs=state
        # )

    return ui


def prepare():
    os.makedirs(embedding.EMBEDDINGS_DIR, exist_ok=True)


if __name__ == "__main__":
    prepare()
    ui = create_ui()
    ui.queue(concurrency_count=4)
    ui.launch()
