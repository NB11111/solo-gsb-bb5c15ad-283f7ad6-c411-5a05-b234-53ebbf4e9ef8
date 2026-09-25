import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mailtrace.parsing import split_batch_headers, thread_analysis  # noqa: E402
from mailtrace.parsing.sample import SAMPLE_HEADERS_THREAD  # noqa: E402
from mailtrace.parsing.thread import extract_message_ids  # noqa: E402

try:  # Flask 相关测试在未安装 Flask 时跳过（解析核心仅需标准库）
    from app import create_app
except ImportError:  # pragma: no cover
    create_app = None


def _mail(index, *, mid=None, irt=None, refs=None, subject=None, extra=""):
    """构造一封仅含会话字段的最小头部。"""
    lines = []
    if mid is not None:
        lines.append(f"Message-ID: <{mid}>")
    if irt is not None:
        lines.append(f"In-Reply-To: {irt}")
    if refs is not None:
        lines.append(f"References: {refs}")
    if subject is not None:
        lines.append(f"Subject: {subject}")
    lines.append(extra.rstrip("\n"))
    return "\n".join(line for line in lines if line is not None)


class ExtractIdTests(unittest.TestCase):
    def test_extracts_angle_bracket_ids_in_order_dedup(self):
        value = "<a@x> text <b@y> <a@x> <c@z>"
        self.assertEqual(extract_message_ids(value), ["a@x", "b@y", "c@z"])

    def test_bare_tokens_outside_brackets_ignored(self):
        self.assertEqual(extract_message_ids("not-an-id <a@x> trailing"), ["a@x"])
        self.assertEqual(extract_message_ids("plain-no-brackets at all"), [])

    def test_empty_and_nested_like(self):
        self.assertEqual(extract_message_ids(""), [])
        self.assertEqual(extract_message_ids("<>"), [])
        # 内层最先匹配，尖括号不作为合法 ID 字符保留
        self.assertEqual(extract_message_ids("<<a@x>>"), ["a@x"])

    def test_folded_value_is_already_unfolded_by_parser(self):
        raw = (
            "References: <a@x>\n"
            "\t<b@y>\n"
            "\t<c@z>\n"
            "Message-ID: <d@x>\n"
        )
        from mailtrace.parsing.parser import parse_header_blocks

        blocks = {b.name.lower(): b for b in parse_header_blocks(raw)}
        self.assertEqual(
            extract_message_ids(blocks["references"].value), ["a@x", "b@y", "c@z"]
        )


class ThreadStructureTests(unittest.TestCase):
    def test_in_reply_to_single_id_is_parent(self):
        mails = [
            _mail(1, mid="a@x", subject="root"),
            _mail(2, mid="b@x", irt="<a@x>", subject="reply"),
        ]
        r = thread_analysis(mails)
        self.assertEqual(r["edge_count"], 1)
        self.assertEqual(r["tree_count"], 1)
        self.assertEqual(r["orphan_count"], 0)
        rows = r["rows"]
        self.assertEqual([row["index"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["root_index"], 1)
        self.assertEqual(rows[0]["depth"], 0)
        self.assertIsNone(rows[0]["parent_index"])
        self.assertEqual(rows[1]["root_index"], 1)
        self.assertEqual(rows[1]["depth"], 1)
        self.assertEqual(rows[1]["parent_index"], 1)
        msg2 = r["messages"][1]
        self.assertEqual(msg2["relation"], "linked")
        self.assertEqual(msg2["parent_index"], 1)
        self.assertEqual(msg2["basis"], "in_reply_to")

    def test_missing_in_reply_to_falls_back_to_last_reference(self):
        mails = [
            _mail(1, mid="a@x"),
            _mail(2, mid="b@x", irt="<a@x>"),
            # 没有 In-Reply-To 字段：References 有两项，取末项 b@x
            _mail(3, mid="c@x", refs="<a@x> <b@x>"),
        ]
        r = thread_analysis(mails)
        msg3 = next(m for m in r["messages"] if m["index"] == 3)
        self.assertEqual(msg3["parent_index"], 2)
        self.assertEqual(msg3["basis"], "references")
        self.assertEqual(msg3["parent_id"], "b@x")

    def test_empty_in_reply_to_field_falls_back_to_references(self):
        mails = [
            _mail(1, mid="a@x"),
            _mail(2, mid="b@x", irt="(no bracket id here)",
                  refs="<a@x>"),
        ]
        r = thread_analysis(mails)
        msg2 = r["messages"][1]
        self.assertEqual(msg2["relation"], "linked")
        self.assertEqual(msg2["parent_index"], 1)
        self.assertEqual(msg2["basis"], "references")

    def test_references_last_item_takes_precedence_over_earlier(self):
        # References 末项是批次内最近的父（b@x），前项 a@x 只是链路上的祖先
        mails = [
            _mail(1, mid="a@x"),
            _mail(2, mid="b@x", irt="<a@x>"),
            _mail(3, mid="d@x", refs="<a@x> <b@x>"),
        ]
        r = thread_analysis(mails)
        msg3 = next(m for m in r["messages"] if m["index"] == 3)
        self.assertEqual(msg3["parent_index"], 2)
        self.assertEqual(msg3["parent_id"], "b@x")
        # 不产生 d@x -> a@x 的越级连边
        self.assertEqual(r["edge_count"], 2)

    def test_external_parent_is_tree_root_and_listed_once(self):
        mails = [
            _mail(1, mid="a@x", irt="<outside@ext>"),
            _mail(2, mid="b@x", irt="<outside@ext>"),
        ]
        r = thread_analysis(mails)
        self.assertEqual(r["external_count"], 1)
        ext = r["external_refs"][0]
        self.assertEqual(ext["message_id"], "outside@ext")
        self.assertEqual(ext["referenced_by"], [1, 2])
        self.assertEqual(r["tree_count"], 2)  # 两封都是外部引用根（均无子）
        root_rows = [row for row in r["rows"] if row["depth"] == 0]
        self.assertEqual([row["index"] for row in root_rows], [1, 2])
        for row in root_rows:
            self.assertEqual(row["external_parent_id"], "outside@ext")
        for m in r["messages"]:
            self.assertEqual(m["relation"], "external")
            self.assertTrue(m["parent_external"])

    def test_orphan_is_root_without_children(self):
        r = thread_analysis([_mail(1, mid="lonely@x")])
        self.assertEqual(r["tree_count"], 0)
        self.assertEqual(r["orphan_count"], 1)
        self.assertEqual(r["orphans"][0]["index"], 1)
        self.assertEqual(r["orphans"][0]["relation"], "root")

    def test_no_message_id_is_not_an_error_but_cannot_be_parent(self):
        mails = [
            _mail(1, mid=None, subject="no id"),
            _mail(2, mid="b@x", refs="<a@x>"),  # a@x 批次外
        ]
        r = thread_analysis(mails)
        self.assertNotIn("error", r)
        no_id = r["messages"][0]
        self.assertEqual(no_id["relation"], "no_id")
        self.assertEqual(r["id_count"], 1)
        # #2 引用的 a@x 既不在批次内（#1 没有 ID），算外部引用
        self.assertEqual(r["messages"][1]["relation"], "external")

    def test_multiple_in_reply_to_ids_is_ambiguous_no_fallback(self):
        mails = [
            _mail(1, mid="a@x"),
            _mail(2, mid="b@x"),
            _mail(3, mid="c@x", irt="<a@x> <b@x>", refs="<a@x>"),
        ]
        r = thread_analysis(mails)
        msg3 = next(m for m in r["messages"] if m["index"] == 3)
        self.assertEqual(msg3["relation"], "ambiguous")
        self.assertIn("multiple_in_reply_to", msg3["ambiguity_codes"])
        # 明确不回退 References：没有任何连边涉及 #3
        self.assertEqual(r["edge_count"], 0)
        # #1、#2 为无根无子的普通邮件（孤立）；#3 进歧义桶而非孤立桶
        self.assertEqual(r["tree_count"], 0)
        self.assertEqual(r["orphan_count"], 2)
        self.assertEqual([m["index"] for m in r["ambiguous_messages"]], [3])

    def test_duplicate_message_id_across_messages_marks_both(self):
        mails = [
            _mail(1, mid="dup@x"),
            _mail(2, mid="dup@x"),
        ]
        r = thread_analysis(mails)
        self.assertEqual(
            r["duplicate_id_sets"],
            [{"message_id": "dup@x", "owner_indexes": [1, 2]}],
        )
        for m in r["messages"]:
            self.assertEqual(m["relation"], "ambiguous")
            self.assertIn("duplicate_message_id", m["ambiguity_codes"])
        self.assertEqual(r["edge_count"], 0)

    def test_child_pointing_to_duplicate_id_is_ambiguous(self):
        mails = [
            _mail(1, mid="dup@x"),
            _mail(2, mid="dup@x"),
            _mail(3, mid="c@x", irt="<dup@x>"),
        ]
        r = thread_analysis(mails)
        msg3 = r["messages"][2]
        self.assertEqual(msg3["relation"], "ambiguous")
        self.assertIn("duplicate_message_id", msg3["ambiguity_codes"])
        self.assertEqual(r["edge_count"], 0)

    def test_multiple_message_ids_in_one_field_is_ambiguous(self):
        raw = "Message-ID: <a@x> <b@x>\nSubject: two ids in one field\n"
        r = thread_analysis([raw])
        m = r["messages"][0]
        self.assertIsNone(m["message_id"])
        self.assertEqual(m["message_id_all"], ["a@x", "b@x"])
        self.assertEqual(m["relation"], "ambiguous")
        self.assertIn("duplicate_message_id", m["ambiguity_codes"])

    def test_two_message_id_fields_is_ambiguous(self):
        raw = (
            "Message-ID: <a@x>\n"
            "Message-ID: <b@x>\n"
            "Subject: two mid fields\n"
        )
        r = thread_analysis([raw])
        m = r["messages"][0]
        self.assertEqual(m["message_id_field_count"], 2)
        self.assertEqual(m["message_id_positions"], [1, 2])
        self.assertEqual(m["relation"], "ambiguous")

    def test_two_node_cycle_drops_both_edges(self):
        mails = [
            _mail(1, mid="a@x", irt="<b@x>"),
            _mail(2, mid="b@x", irt="<a@x>"),
        ]
        r = thread_analysis(mails)
        self.assertEqual(r["edge_count"], 0)
        for m in r["messages"]:
            self.assertEqual(m["relation"], "ambiguous")
            self.assertIn("cycle", m["ambiguity_codes"])
        self.assertEqual(r["tree_count"], 0)
        self.assertEqual(r["orphan_count"], 0)

    def test_self_loop_drops_edge(self):
        r = thread_analysis([_mail(1, mid="a@x", irt="<a@x>")])
        m = r["messages"][0]
        self.assertEqual(m["relation"], "ambiguous")
        self.assertIn("cycle", m["ambiguity_codes"])
        self.assertEqual(r["edge_count"], 0)

    def test_three_node_cycle_with_incoming_branch(self):
        # a <- b <- c <- a 成环；d 正常引用 a（挂在环上）。
        mails = [
            _mail(1, mid="a@x", irt="<c@x>"),
            _mail(2, mid="b@x", irt="<a@x>"),
            _mail(3, mid="c@x", irt="<b@x>"),
            _mail(4, mid="d@x", irt="<a@x>"),
        ]
        r = thread_analysis(mails)
        # 环上三封的出边全部删除；#4 -> #1 的入边保留
        self.assertEqual(r["edge_count"], 1)
        for idx in (1, 2, 3):
            self.assertEqual(r["messages"][idx - 1]["relation"], "ambiguous")
        msg4 = r["messages"][3]
        self.assertEqual(msg4["relation"], "linked")
        self.assertEqual(msg4["parent_index"], 1)
        # #1 有子 #4，仍作为一棵树的根展示（#1 本身带歧义标记）
        self.assertEqual(r["tree_count"], 1)

    def test_normal_tree_not_affected_by_ambiguous_mail(self):
        mails = [
            _mail(1, mid="a@x"),
            _mail(2, mid="b@x", irt="<a@x>"),
            _mail(3, mid="c@x", irt="<a@x> <b@x>"),  # 歧义
        ]
        r = thread_analysis(mails)
        self.assertEqual(r["edge_count"], 1)
        rows = r["rows"]
        self.assertEqual([row["index"] for row in rows], [1, 2])
        self.assertEqual([row["depth"] for row in rows], [0, 1])
        self.assertEqual(rows[1]["parent_index"], 1)

    def test_case_insensitive_field_names_and_folding(self):
        # 字段名大小写不敏感；In-Reply-To 折行续行展开后仍能取出尖括号 ID
        raw = (
            "message-id: <child@x>\n"
            "IN-REPLY-TO: pre\n"
            "\t<a@x> post\n"
        )
        mails = [_mail(1, mid="a@x"), raw]
        r = thread_analysis(mails)
        self.assertEqual(r["messages"][1]["parent_index"], 1)

    def test_unparseable_header_reports_error_but_others_continue(self):
        mails = [_mail(1, mid="a@x"), "   ", _mail(3, mid="b@x", irt="<a@x>")]
        r = thread_analysis(mails)
        self.assertNotIn("error", r)
        self.assertEqual(r["message_count"], 3)
        self.assertEqual(r["parsed_count"], 2)
        self.assertEqual(r["message_errors"],
                         [{"index": 2, "error": "该头部无法解析为任何邮件字段"}])
        self.assertEqual(r["edge_count"], 1)

    def test_all_unparseable_is_error(self):
        r = thread_analysis(["   ", "\n"])
        self.assertIn("error", r)
        self.assertEqual(r["parsed_count"], 0)
        self.assertEqual(len(r["message_errors"]), 2)

    def test_empty_list_is_error(self):
        self.assertIn("error", thread_analysis([]))


class SampleThreadTests(unittest.TestCase):
    def setUp(self):
        self.parts = split_batch_headers(SAMPLE_HEADERS_THREAD)

    def test_sample_splits_into_twelve(self):
        self.assertEqual(len(self.parts), 12)

    def test_sample_overall_counts(self):
        r = thread_analysis(self.parts)
        self.assertEqual(r["parsed_count"], 12)
        self.assertEqual(r["edge_count"], 3)      # 1->2, 2->3, 2->4
        self.assertEqual(r["tree_count"], 2)      # 主树 + 外部引用根 #5
        self.assertEqual(r["orphan_count"], 1)    # #6
        self.assertEqual(r["external_count"], 1)  # #5 的批次外父 ID
        # #7 多 ID、#8/#9 重复 MID、#10 指向重复 ID、#11/#12 成环
        self.assertEqual(r["ambiguous_count"], 6)

    def test_sample_main_tree_shape(self):
        r = thread_analysis(self.parts)
        rows = [row for row in r["rows"] if row["root_index"] == 1]
        # 深度优先行序：1（根）→ 2 → 3、4（2 的子，按批次序号）
        self.assertEqual([row["index"] for row in rows], [1, 2, 3, 4])
        self.assertIsNone(rows[0]["external_parent_id"])
        depths = {row["index"]: row["depth"] for row in rows}
        self.assertEqual(depths, {1: 0, 2: 1, 3: 2, 4: 2})
        parents = {row["index"]: row["parent_index"] for row in rows}
        self.assertIsNone(parents[1])
        self.assertEqual(parents[2], 1)
        self.assertEqual(parents[3], 2)
        self.assertEqual(parents[4], 2)
        # #4 通过 References 末项连到 #2
        msg4 = next(m for m in r["messages"] if m["index"] == 4)
        self.assertEqual(msg4["basis"], "references")

    def test_sample_external_ref(self):
        r = thread_analysis(self.parts)
        ext = r["external_refs"]
        self.assertEqual(ext[0]["message_id"], "message-outside-batch@external.example")
        self.assertEqual(ext[0]["referenced_by"], [5])

    def test_sample_ambiguity_codes(self):
        r = thread_analysis(self.parts)
        codes = {m["index"]: m["ambiguity_codes"] for m in r["messages"]}
        self.assertEqual(codes[7], ["multiple_in_reply_to"])
        self.assertEqual(codes[8], ["duplicate_message_id"])
        self.assertEqual(codes[9], ["duplicate_message_id"])
        self.assertEqual(codes[10], ["duplicate_message_id"])
        self.assertEqual(codes[11], ["cycle"])
        self.assertEqual(codes[12], ["cycle"])


class LongChainTests(unittest.TestCase):
    """超长单链（逐封 In-Reply-To 上一封）：扁平行迭代展开，不触发递归上限。"""

    @staticmethod
    def _chain(n):
        mails = [_mail(1, mid="m1@x", subject="chain root")]
        for i in range(2, n + 1):
            mails.append(_mail(i, mid=f"m{i}@x", irt=f"<m{i - 1}@x>"))
        return mails

    def test_long_single_chain_expands_iteratively(self):
        n = 1500  # 超过 Python 默认递归上限（1000）
        r = thread_analysis(self._chain(n))
        self.assertNotIn("error", r)
        self.assertEqual(r["parsed_count"], n)
        self.assertEqual(r["edge_count"], n - 1)
        self.assertEqual(r["tree_count"], 1)
        self.assertEqual(r["orphan_count"], 0)
        rows = r["rows"]
        self.assertEqual(len(rows), n)
        # 单链的深度优先行序即输入序；行内根序号 / 父序号 / 深度完整
        for pos, row in enumerate(rows, start=1):
            self.assertEqual(row["index"], pos)
            self.assertEqual(row["root_index"], 1)
            self.assertEqual(row["depth"], pos - 1)
            self.assertEqual(
                row["parent_index"], None if pos == 1 else pos - 1
            )
        self.assertEqual(rows[0]["relation"], "root")
        self.assertTrue(all(row["relation"] == "linked" for row in rows[1:]))

    def test_long_chain_with_side_branches_keeps_dfs_order(self):
        # 主链 1200 封，另在第 2 封下挂两个分支（批次内第 1201、1202 封），
        # 验证行序仍为深度优先
        mails = self._chain(1200)
        mails.append(_mail(1201, mid="s1@x", irt="<m2@x>"))
        mails.append(_mail(1202, mid="s2@x", irt="<m2@x>"))
        r = thread_analysis(mails)
        rows = r["rows"]
        self.assertEqual(len(rows), 1202)
        # 第 2 封之后先展开主链（子序号 3 < 1201 < 1202），分支排在主链之后
        self.assertEqual(rows[1]["index"], 2)
        self.assertEqual(rows[2]["index"], 3)
        self.assertEqual([row["index"] for row in rows[-2:]], [1201, 1202])
        self.assertTrue(all(row["depth"] == 2 for row in rows[-2:]))
        self.assertTrue(all(row["parent_index"] == 2 for row in rows[-2:]))


@unittest.skipIf(create_app is None, "未安装 Flask，跳过 API 测试")
class ThreadApiTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_api_thread_ok(self):
        parts = split_batch_headers(SAMPLE_HEADERS_THREAD)
        resp = self.client.post("/api/thread-relations", json={"headers": parts})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["parsed_count"], 12)
        self.assertEqual(data["tree_count"], 2)
        self.assertIn("rows", data)
        self.assertIn("messages", data)
        self.assertIn("ambiguous_messages", data)
        # 页面与 API 共用同一行序：主树 1→2→3→4，外部引用根 #5
        self.assertEqual([row["index"] for row in data["rows"]], [1, 2, 3, 4, 5])

    def test_api_thread_long_chain_no_recursion_error(self):
        n = 1200
        mails = ["Message-ID: <m1@x>\nSubject: chain root\n"]
        mails += [
            f"Message-ID: <m{i}@x>\nIn-Reply-To: <m{i - 1}@x>\n"
            for i in range(2, n + 1)
        ]
        resp = self.client.post("/api/thread-relations", json={"headers": mails})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["tree_count"], 1)
        self.assertEqual(len(data["rows"]), n)
        self.assertEqual(data["rows"][-1]["depth"], n - 1)
        self.assertEqual(data["rows"][-1]["parent_index"], n - 1)

    def test_api_thread_accepts_single_header(self):
        resp = self.client.post(
            "/api/thread-relations",
            json={"headers": ["Message-ID: <a@x>\nSubject: one\n"]},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["orphan_count"], 1)

    def test_api_thread_requires_json(self):
        resp = self.client.post("/api/thread-relations", data={"headers": "x"})
        self.assertEqual(resp.status_code, 400)

    def test_api_thread_rejects_non_list(self):
        resp = self.client.post("/api/thread-relations", json={"headers": "nope"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("字符串数组", resp.get_json()["error"])

    def test_api_thread_rejects_non_string_items(self):
        resp = self.client.post(
            "/api/thread-relations", json={"headers": ["Message-ID: <a@x>", 1]}
        )
        self.assertEqual(resp.status_code, 400)

    def test_api_thread_rejects_empty_list_and_blank_items(self):
        self.assertEqual(
            self.client.post("/api/thread-relations", json={"headers": []}).status_code,
            400,
        )
        resp = self.client.post(
            "/api/thread-relations", json={"headers": ["Message-ID: <a@x>", " "]}
        )
        self.assertEqual(resp.status_code, 400)

    def test_form_page_renders_tree(self):
        resp = self.client.post(
            "/thread-relations", data={"headers_thread": SAMPLE_HEADERS_THREAD}
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("邮件会话关系结果", html)
        self.assertIn("会话树", html)
        self.assertIn("孤立邮件", html)
        self.assertIn("外部引用", html)
        self.assertIn("逐封关联依据", html)

    def test_form_page_no_delimiter_single(self):
        resp = self.client.post(
            "/thread-relations",
            data={"headers_thread": "Message-ID: <a@x>\nSubject: single\n"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("邮件会话关系结果", resp.get_data(as_text=True))

    def test_form_page_empty_input_shows_banner(self):
        resp = self.client.post("/thread-relations", data={"headers_thread": "   "})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("未识别到任何邮件头部", resp.get_data(as_text=True))

    def test_form_page_long_chain_renders_without_recursion(self):
        n = 1200
        mails = ["Message-ID: <m1@x>\nSubject: chain root"]
        mails += [
            f"Message-ID: <m{i}@x>\nIn-Reply-To: <m{i - 1}@x>"
            for i in range(2, n + 1)
        ]
        raw = "\n===== 下一封邮件 =====\n".join(mails)
        resp = self.client.post("/thread-relations", data={"headers_thread": raw})
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("邮件会话关系结果", html)
        self.assertIn(f"深度 {n - 1}", html)  # 最深的一行也完整渲染


if __name__ == "__main__":
    unittest.main()
