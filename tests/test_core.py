import tempfile,unittest,json
from pathlib import Path
import torch
from torch import nn
from lora import LoRALinear,configure_lora,adapter_state,restore_adapter
from metrics import score_answer
from data import check_document_splits,load_manifest
from modeling import tiny_llava,tiny_batch
from batching import build_example,answer_loss

class CoreTests(unittest.TestCase):
    def test_anls_multi_answer_threshold_and_normalization(self):
        self.assertEqual(score_answer('  ABC  ',['wrong','abc']),{'anls':1.,'exact_match':1.})
        self.assertEqual(score_answer('ab',['ac'])['anls'],0.)
        self.assertAlmostEqual(score_answer('invoice',['invoic'])['anls'],6/7)
    def test_document_leakage_rejected(self):
        with self.assertRaises(ValueError):check_document_splits([{'document_id':'x','split':'train'}],[{'document_id':'x','split':'val'}])
    def test_zero_init_lora_and_frozen_base(self):
        base=nn.Linear(8,4);model=LoRALinear(base,2,4,0);x=torch.randn(3,8)
        torch.testing.assert_close(model(x),base(x),rtol=0,atol=0)
        model(x).square().sum().backward()
        self.assertIsNone(base.weight.grad);self.assertGreater(model.b.grad.abs().sum().item(),0)
    def test_real_llava_image_token_contract(self):
        model=tiny_llava();batch=tiny_batch();result=model(**batch,use_cache=False)
        self.assertEqual(tuple(result.logits.shape),(1,10,64));self.assertTrue(torch.isfinite(result.loss))
        bad={**batch,'input_ids':batch['input_ids'].clone()};bad['input_ids'][0,1]=3
        with self.assertRaises(ValueError):model(**bad,use_cache=False)
    def test_adapter_trainability_and_reload(self):
        model=tiny_llava();configure_lora(model,2,4,0,True)
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        self.assertTrue(names);self.assertFalse(any('vision_tower' in n for n in names))
        self.assertTrue(all(n.endswith(('.a','.b')) or 'multi_modal_projector' in n for n in names))
        state=adapter_state(model);restore_adapter(model,state)
        with self.assertRaises(ValueError):restore_adapter(model,{})
    def test_answer_masking_and_prefix_rejection(self):
        # Token fixtures test the collator's boundary checks, not model QA quality.
        class Tokenizer:
            eos_token='<eos>'
            def convert_tokens_to_ids(self,token):return 63
        class Processor:
            tokenizer=Tokenizer();image_token='<image>'
            def apply_chat_template(self,messages,**kwargs):return 'full' if len(messages)>1 else 'prefix'
            def __call__(self,text,images,**kwargs):
                ids=torch.tensor([[1,63,10]] if text=='prefix' else [[1,63,10,20,2]])
                return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.png';Image.new('RGB',(28,28)).save(path)
            row={'image_path':str(path),'question':'q','answers':['a']}
            batch=build_example(Processor(),row)
            self.assertEqual(batch['labels'].tolist(),[[-100,-100,-100,20,2]])
            with self.assertRaises(ValueError):build_example(Processor(),row,max_length=4)

    def test_suffix_loss_matches_full_causal_loss(self):
        model=tiny_llava().eval();batch=tiny_batch()
        full=model(**batch,use_cache=False).loss
        torch.testing.assert_close(answer_loss(model,batch),full,rtol=1e-5,atol=1e-6)
